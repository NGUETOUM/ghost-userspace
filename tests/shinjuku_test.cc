#include <functional>
#include <iostream>
#include <vector>
#include <unistd.h>


#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <sys/types.h>
#include <stdbool.h>


#include "absl/synchronization/notification.h"
#include "experiments/shared/ghost.h"
#include "experiments/shared/thread_pool.h"
#include "lib/base.h"
#include "lib/ghost.h"



/**************************STDDEFINE.H START****************************/

//#define TIMING

/* Debug printf */
#define dprintf(...) fprintf(stdout, __VA_ARGS__)

/* Wrapper to check for errors */
#define CHECK_ERROR(a)                                       \
   if (a)                                                    \
   {                                                         \
      perror("Error at line\n\t" #a "\nSystem Msg");         \
      assert ((a) == 0);                                     \
   }

static inline void *MALLOC(size_t size)
{
   void * temp = malloc(size);
   assert(temp);
   return temp;
}

static inline void *CALLOC(size_t num, size_t size)
{
   void * temp = calloc(num, size);
   assert(temp);
   return temp;
}

static inline void *REALLOC(void *ptr, size_t size)
{
   void * temp = realloc(ptr, size);
   assert(temp);
   return temp;
}

static inline char *GETENV(char *envstr)
{
   char *env = getenv(envstr);
   if (!env) return "0";
   else return env;
}

#define GET_TIME(start, end, duration)                                     \
   duration.tv_sec = (end.tv_sec - start.tv_sec);                         \
   if (end.tv_nsec >= start.tv_nsec) {                                     \
      duration.tv_nsec = (end.tv_nsec - start.tv_nsec);                   \
   }                                                                       \
   else {                                                                  \
      duration.tv_nsec = (1000000000L - (start.tv_nsec - end.tv_nsec));   \
      duration.tv_sec--;                                                   \
   }                                                                       \
   if (duration.tv_nsec >= 1000000000L) {                                  \
      duration.tv_sec++;                                                   \
      duration.tv_nsec -= 1000000000L;                                     \
   }

static inline unsigned int time_diff (
    struct timeval *end, struct timeval *begin)
{
#ifdef TIMING
    uint64_t result;

    result = end->tv_sec - begin->tv_sec;
    result *= 1000000;     /* usec */
    result += end->tv_usec - begin->tv_usec;

    return result;
#else
    return 0;
#endif
}

static inline void get_time (struct timeval *t)
{
#ifdef TIMING
    gettimeofday (t, NULL);
#endif
}

/*****************************STDDEFINE.H END**************************************/


/*******************************MAP_RECUDE.H START*********************************/

/* Standard data types for the function arguments and results */

/* Argument to map function. This is specified by the splitter function.
 * length - number of elements of data. The default splitter function gives
            length in terms of the # of elements of unit_size bytes.
 * data - data to process of a user defined type
 */
typedef struct
{
   intptr_t length;
   void *data;
} map_args_t;

/* Single element of result
 * key - pointer to the key
 * val - pointer to the value
 */
typedef struct
{
   void *key;
   void *val;
} keyval_t;

/* List of results
 * length - number of key value pairs
 * data - array of key value pairs
 */
typedef struct
{
   int length;
   keyval_t *data;
} final_data_t;

/* Scheduler function pointer type definitions */

/* Map function takes in map_args_t, as supplied by the splitter
 * and emit_intermediate() should be called on any key value pairs
 * in the intermediate result set.
 */
typedef void (*map_t)(map_args_t *);

struct iterator_t;
typedef struct iterator_t iterator_t;
int iter_next (iterator_t *itr, void **);
int iter_size (iterator_t *itr);

/* Reduce function takes in a key pointer, a list of value pointers, and a
 * length of the list. emit() should be called on any key value pairs
 * in the result set.
 */
typedef void (*reduce_t)(void *, iterator_t *itr);

/* Combiner function takes in an iterator for a particular key,
 * and returns a reduced value. The operation should be identical to the
 * reduce function, except that this function returns the reduced value
 * directly. */
typedef void *(*combiner_t)(iterator_t *itr);

/* Splitter function takes in a pointer to the input data, an interger of
 * the number of bytes requested, and an uninitialized pointer to a
 * map_args_t pointer. The result is stored in map_args_t. The splitter
 * should return 1 if the result is valid or 0 if there is no more data.
 */
typedef int (*splitter_t)(void *, int, map_args_t *);

/* Locator function takes in a pointer to map_args_t, and returns
 * the memory address where this map task would be heavily accessing.
 * The runtime would then try to dispatch the task to a thread that
 * is nearby the physical memory that backs the address. */
typedef void* (*locator_t)(map_args_t *);

/* Partition function takes in the number of reduce tasks, a pointer to
 * a key, and the lendth of the key in bytes. It assigns a key to a reduce task.
 * The value returned is the # of the reduce task where the key will be processed.
 * This value should be the same for all keys that are equal.
 */
typedef int (*partition_t)(int, void *, int);

/* key_cmp(key1, key2) returns:
 *   0 if key1 == key2
 *   + if key1 > key2
 *   - if key1 < key2
 */
typedef int (*key_cmp_t)(const void *, const void*);

/* The arguments to operate the runtime. */
typedef struct
{
    void * task_data;           /* The data to run MapReduce on.
                                 * If splitter is NULL, this should be an array. */
    off_t data_size;            /* Total # of bytes of data */
    int unit_size;              /* # of bytes for one element
                                 * (if necessary, on average) */

    map_t map;                  /* Map function pointer, must be user defined */
    reduce_t reduce;            /* If NULL, identity reduce function is used,
                                 * which emits a keyval pair for each val. */
    combiner_t combiner;        /* If NULL, no combiner would be called. */
    splitter_t splitter;        /* If NULL, the array splitter is used.*/
    locator_t locator;          /* If NULL, no locality based optimization is
                                   performed. */
    key_cmp_t key_cmp;          /* Key comparison function.
                                   Must be user defined.*/

    final_data_t *result;       /* Pointer to output data.
                                 * Must be allocated by user */

    /*** Optional arguments must be zero if not used ***/
    partition_t partition;      /* Default partition function is a
                                 * hash function */

    /* Creates one emit queue for each reduce task,
    * instead of per reduce thread. This improves
    * time to emit if data is emitted in order,
    * but can increase merge time. */
    bool use_one_queue_per_task;

    int L1_cache_size;     /* Size of L1 cache in bytes */
    int num_map_threads;   /* # of threads to run map tasks on.
                                 * Default is one per processor */
    int num_reduce_threads;     /* # of threads to run reduce tasks on.
    * Default is one per processor */
    int num_merge_threads;      /* # of threads to run merge tasks on.
    * Default is one per processor */
    int num_procs;              /* Maximum number of processors to use. */

    int proc_offset;            /* number of procs to skip for thread binding */
                                /* (useful if you have multiple MR's running
                                 *  and you don't want them binding to the same
                                 *  hardware thread). */

    float key_match_factor;     /* Magic number that describes the ratio of
    * the input data size to the output data size.
    * This is used as a hint. */
} map_reduce_args_t;

/* Runtime defined functions. */

/* MapReduce initialization function. Called once per process. */
int map_reduce_init ();

/* MapReduce finalization function. Called once per process. */
int map_reduce_finalize ();

/* The main MapReduce engine. This is the function called by the application.
 * It is responsible for creating and scheduling all map and reduce tasks, and
 * also organizes and maintains the data which is passed from application to
 * map tasks, map tasks to reduce tasks, and reduce tasks back to the
 * application. Results are stored in args->result. A return value less than zero
 * represents an error. This function is not thread safe.
 */
int map_reduce (map_reduce_args_t * args);

/* This should be called from the map function. It stores a key with key_size
 * bytes and a value in the intermediate queues for processing by the reduce
 * task. The runtime will call partiton function to assign the key to a
 * reduce task.
 */
void emit_intermediate(void *key, void *val, int key_size);

/* This should be called from the reduce function. It stores a key and a value
 * in the reduce queue. This will be in the final result array.
 */
void emit(void *key, void *val);

/* This is the built in partition function which is a hash.  It is global so
 * the user defined partition function can call it.
 */
int default_partition(int reduce_tasks, void* key, int key_size);


/*************************************MAP_REDUCE.H END********************************/

//INITIALIZE GLOBALS VARIABLES TO MANAGE THREADS ON A PRIOTABLE

ghost_test::Ghost ghost_(7, 7);
ghost_test::ExperimentThreadPool thread_pool_(7);
std::vector<ghost::GhostThread::KernelScheduler> kernelSchedulers(
    7, ghost::GhostThread::KernelScheduler::kGhost);
std::vector<std::function<void(uint32_t)>> threadWork;

absl::Notification printed_0;
absl::Notification printed_1;
absl::Notification printed_2;
absl::Notification printed_3;
absl::Notification printed_4;
absl::Notification printed_5;
absl::Notification printed_6;
absl::Notification wait_0;
absl::Notification wait_1;
absl::Notification wait_2;
absl::Notification wait_3;
absl::Notification wait_4;
absl::Notification wait_5;
absl::Notification wait_6;

namespace {
// We do not need a different class of service (e.g., different expected
// runtimes, different QoS (Quality-of-Service) classes, etc.) across workers in
// our experiments. Furthermore, all workers are ghOSt one-shots. Thus, put all
// worker sched items in the same work class.
static constexpr uint32_t kWorkClassIdentifier = 0;
}  // namespace


#define DEFAULT_DISP_NUM 10
#define START_ARRAY_SIZE 2000

typedef struct {
	char* word;
	int count;
} wc_count_t;

typedef struct {
   long fpos;
   long flen;
   char *fdata;
   int unit_size;
} wc_data_t;

typedef struct
{
   int length;
   void *data;
   int t_num;
} t_args_t;

enum {
   IN_WORD,
   NOT_IN_WORD
};

typedef struct
{
   int length1;
   int length2;
   int length_out_pos;
   wc_count_t *data1;
   wc_count_t *data2;
   wc_count_t *out;
} merge_data_t;

typedef struct {
	void* base;
	size_t num_elems;
	size_t width;
	int (*compar)(const void *, const void *);
} sort_args;

wc_count_t** words;
int* use_len;
int* length;

void wordcount_map(absl::Notification* printed, absl::Notification* wait, t_args_t* args_in/*, merge_data_t* m_args*/);
void wordcount_reduce(char* word, int len) ;
/**********************************************************************/


/** mystrcmp()
 *  Comparison function to compare 2 words
 */
inline int mystrcmp(const void *s1, const void *s2)
{
   return strcmp((const char *)s1, (const char *) s2);
}

/** wordcount_cmp()
 *  Comparison function to compare 2 words
 */
int wordcount_cmp(const void *v1, const void *v2)
{
   wc_count_t* w1 = (wc_count_t*)v1;
   wc_count_t* w2 = (wc_count_t*)v2;

   int i1 = w1->count;
   int i2 = w2->count;

   if (i1 < i2) return 1;
   else if (i1 > i2) return -1;
   else return 0;
}

/** wordcount_splitter()
 *  Memory map the file and divide file on a word border i.e. a space.
 *	Assign each portion of the file to a thread
 */
void wordcount_splitter(void *data_in)
{
   int i,num_procs;

   num_procs = 7;

   wc_data_t * data = (wc_data_t *)data_in;

   words = (wc_count_t**)malloc(num_procs*sizeof(wc_count_t*));
   length = (int*)malloc(num_procs*sizeof(int));
   use_len = (int*)malloc(num_procs*sizeof(int));

   int req_bytes = data->flen / num_procs;


   wc_count_t** mwords = (wc_count_t**)malloc(num_procs*sizeof(wc_count_t*));

   for(i = 0; i < num_procs; i++)
   {
      words[i] = (wc_count_t*)malloc(START_ARRAY_SIZE*sizeof(wc_count_t));
      length[i] = START_ARRAY_SIZE;
      use_len[i] = 0;

     t_args_t* out = (t_args_t*)malloc(sizeof(t_args_t));
	   out->data = &data->fdata[data->fpos];

	   int available_bytes = data->flen - data->fpos;
	   if(available_bytes < 0)
		   available_bytes = 0;

      out->t_num = i;
	   out->length = (req_bytes < available_bytes)? req_bytes:available_bytes;

	   // Set the length to end at a space
	   for (data->fpos += (long)out->length;
			data->fpos < data->flen &&
			data->fdata[data->fpos] != ' ' && data->fdata[data->fpos] != '\t' &&
			data->fdata[data->fpos] != '\r' && data->fdata[data->fpos] != '\n';
			data->fpos++, out->length++);

      //merge part

      merge_data_t* m_args = (merge_data_t*)malloc(sizeof(merge_data_t));
      m_args->length1 = use_len[i*2];
        m_args->length2 = use_len[i*2 + 1];
        m_args->length_out_pos = i;
        m_args->data1 = words[i*2];
        m_args->data2 = words[i*2 + 1];
        int tlen = m_args->length1 + m_args->length2;
        mwords[i] = (wc_count_t*)malloc(tlen*sizeof(wc_count_t));
        m_args->out = mwords[i];
     if(i == 0){
      threadWork.push_back(std::bind(&wordcount_map, &printed_0, &wait_0, out/*, m_args*/));
    }else if(i == 1){
      threadWork.push_back(std::bind(&wordcount_map, &printed_1, &wait_1, out/*, m_args*/));
    }else if(i == 2){
      threadWork.push_back(std::bind(&wordcount_map, &printed_2, &wait_2, out/*, m_args*/));
    }else if(i == 3){
      threadWork.push_back(std::bind(&wordcount_map, &printed_3, &wait_3, out/*, m_args*/));
    }else if(i == 4){
      threadWork.push_back(std::bind(&wordcount_map, &printed_4, &wait_4, out/*, m_args*/));
    }else if(i == 5){
      threadWork.push_back(std::bind(&wordcount_map, &printed_5, &wait_5, out/*, m_args*/));
    }else if(i == 6){
      threadWork.push_back(std::bind(&wordcount_map, &printed_6, &wait_6, out/*, m_args*/));
    }

   }

   thread_pool_.Init(kernelSchedulers, threadWork);
   printf("\n string capacity is %d \n", ghost_.table_.hdr()->st_cap);
  for (size_t k = 0; k < thread_pool_.GetGtids().size(); ++k) {
      ghost::sched_item si;
      ghost_.GetSchedItem(k, si);
      si.sid = k;
      si.wcid = kWorkClassIdentifier;
      si.gpid = thread_pool_.GetGtids()[k].id();
      printf("\n thread id %d \n", thread_pool_.GetGtids()[k].id());
      si.flags |= SCHED_ITEM_RUNNABLE;
      ghost_.SetSchedItem(k, si);

      if(k == 0){
      printed_0.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_0.Notify();
    }else if(k == 1){
      printed_1.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_1.Notify();
    }else if(k == 2){
      printed_2.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_2.Notify();
    }else if(k == 3){
      printed_3.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_3.Notify();
    }else if(k == 4){
      printed_4.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_4.Notify();
    }else if(k == 5){
      printed_5.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_5.Notify();
    }else if(k == 6){
      printed_6.WaitForNotification();
      thread_pool_.MarkExit(/*sid=*/ k);
      wait_6.Notify();
    }

  }

    thread_pool_.Join();

}

/** wordcount_map()
 * Go through the allocated portion of the file and count the words
 */
void wordcount_map(absl::Notification* printed, absl::Notification* wait, t_args_t* args_in /*merge_data_t* m_args*/)
{
	t_args_t* args = args_in;

   char *curr_start, curr_ltr;
   int state = NOT_IN_WORD;
   int i;
   assert(args);

   char *data = (char *)(args->data);
   curr_start = data;
   assert(data);

   for (i = 0; i < args->length; i++)
   {
      curr_ltr = toupper(data[i]);
      switch (state)
      {
      case IN_WORD:
         data[i] = curr_ltr;
         if ((curr_ltr < 'A' || curr_ltr > 'Z') && curr_ltr != '\'')
         {
            data[i] = 0;
			//printf("\nthe word is %s\n\n",curr_start);
			wordcount_reduce(curr_start, args->t_num);
            state = NOT_IN_WORD;
         }
      break;

      default:
      case NOT_IN_WORD:
         if (curr_ltr >= 'A' && curr_ltr <= 'Z')
         {
            curr_start = &data[i];
            data[i] = curr_ltr;
            state = IN_WORD;
         }
         break;
      }
   }

   // Add the last word
   if (state == IN_WORD)
   {
			data[args->length] = 0;
			//printf("\nthe word is %s\n\n",curr_start);
			wordcount_reduce(curr_start, args->t_num);
   }


   printed->Notify();
   wait->WaitForNotification();
}

/** wordcount_reduce()
 * Locate the key in the array of word counts and
 * add up the partial sums for each word
 */
void wordcount_reduce(char* word, int t_num)
{
   int cmp=-1, high = use_len[t_num], low = -1, next;

   // Binary search the array to find the key
   while (high - low > 1)
   {
       next = (high + low) / 2;
       cmp = strcmp(word, words[t_num][next].word);
       if (cmp == 0)
       {
          high = next;
          break;
       }
       else if (cmp < 0)
           high = next;
       else
           low = next;
   }

	int pos = high;

   if (pos >= use_len[t_num])
   {
      // at end
      words[t_num][use_len[t_num]].word = word;
	   words[t_num][use_len[t_num]].count = 1;
	   use_len[t_num]++;
	}
   else if (pos < 0)
   {
      // at front
      memmove(&words[t_num][1], words[t_num], use_len[t_num]*sizeof(wc_count_t));
      words[t_num][0].word = word;
	   words[t_num][0].count = 1;
	   use_len[t_num]++;
   }
   else if (cmp == 0)
   {
      // match
      words[t_num][pos].count++;
	}
   else
   {
      // insert at pos
      memmove(&words[t_num][pos+1], &words[t_num][pos], (use_len[t_num]-pos)*sizeof(wc_count_t));
      words[t_num][pos].word = word;
	   words[t_num][pos].count = 1;
	   use_len[t_num]++;
   }

	if(use_len[t_num] == length[t_num])
	{
		length[t_num] *= 2;
	   words[t_num] = (wc_count_t*)realloc(words[t_num],length[t_num]*sizeof(wc_count_t));
	}
}


int main(int argc, char const* argv[]) {

   int fd;
   char * fdata;
   int disp_num;
   struct stat finfo;
   char * fname, * disp_num_str;

   absl::Time start;
   absl::Time end;

   fname = "/home/armel/Downloads/word_count/word_count_datafiles/word_100MB.txt";
   disp_num_str = NULL;

   printf("Wordcount: Running...\n");

   // Read in the file
   CHECK_ERROR((fd = open(fname, O_RDONLY)) < 0);
   // Get the file info (for file length)
   CHECK_ERROR(fstat(fd, &finfo) < 0);
   // Memory map the file
   CHECK_ERROR((fdata = (char *) mmap(0, finfo.st_size + 1,
      PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)) == NULL);

   // Get the number of results to display
   CHECK_ERROR((disp_num = (disp_num_str == NULL) ?
      DEFAULT_DISP_NUM : atoi(disp_num_str)) <= 0);

   // Setup splitter args
   wc_data_t wc_data;
   wc_data.unit_size = 5; // approx 5 bytes per word
   wc_data.fpos = 0;
   wc_data.flen = finfo.st_size;
   wc_data.fdata = fdata;

   dprintf("Wordcount: Calling MapReduce Scheduler Wordcount\n");


   start = absl::Now();

   wordcount_splitter(&wc_data);

   end = absl::Now();


  printf("\n End of Map function \n");

   printf(" The execution time is  %0.2f ms\n", absl::ToDoubleMilliseconds(end - start));


  return 0;
}
