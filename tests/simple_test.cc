#include <functional>
#include <iostream>
#include <vector>
#include <unistd.h>

#include "absl/synchronization/notification.h"
#include "experiments/shared/ghost.h"
#include "experiments/shared/thread_pool.h"
#include "lib/base.h"
#include "lib/ghost.h"

void printHelloWorld(uint32_t num, absl::Notification* printed,
                     absl::Notification* wait) {
    int j = 0;
    std::cout << num << std::endl;
    for(int count = 0; count < 10; count++){
        std::cout << count << std::endl;
        std::cout << "Hello World" << std::endl;
      for(int i = 0; i < 1000000; i++){
        j++;
        j--;
      }
        sleep(0.0001);
    }
  printed->Notify();
  wait->WaitForNotification();
}

int main(int argc, char const* argv[]) {
  ghost_test::Ghost ghost_(4, 4);
  ghost_test::ExperimentThreadPool thread_pool_(4);
  std::vector<ghost::GhostThread::KernelScheduler> kernelSchedulers(
      4, ghost::GhostThread::KernelScheduler::kGhost);;
  std::vector<std::function<void(uint32_t)>> threadWork;
  kernelSchedulers.push_back(ghost::GhostThread::KernelScheduler::kGhost);

  absl::Notification printed;
  absl::Notification wait;
  threadWork.push_back(
      std::bind(printHelloWorld, std::placeholders::_1, &printed, &wait));

  thread_pool_.Init(kernelSchedulers, threadWork);
  ghost::sched_item si;
  ghost_.GetSchedItem(0, si);
  si.sid = 0;
  si.gpid = thread_pool_.GetGtids()[0].id();
  si.flags |= SCHED_ITEM_RUNNABLE;
  ghost_.SetSchedItem(0, si);

  printed.WaitForNotification();
  thread_pool_.MarkExit(/*sid=*/0);
  wait.Notify();
  thread_pool_.Join();

  return 0;
}
