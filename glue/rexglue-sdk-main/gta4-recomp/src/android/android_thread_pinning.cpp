// Keeps the busiest threads of the process on dedicated big cores.
//
// Six threads carry the frame (guest render thread, recorder, command worker,
// and three lighter ones) and the scheduler moves them freely: the guest
// render thread was seen on a little core while the prime core idled, and the
// recorder spent ~30% of each frame off-CPU. Every two seconds this ranks the
// process's threads by CPU time over the last interval, gives the top ones a
// big core each (fastest first) and confines every other thread to the
// remaining cores. Threads are re-ranked continuously, so roles that move
// between threads (loading, cutscenes) follow.
//
// Default is ON (4 threads): tuned for the Dimensity 8500 Ultra (POCO X8 Pro),
// 8x Cortex-A725 in three tiers (1x 3.4 GHz prime, 3x 3.2 GHz, 4x 2.2 GHz).
// The four fastest cores take the four busiest threads, the 2.2 GHz cores take
// everything else. Cores are ranked by max frequency, then cpu_capacity, then
// highest cpu index (the fast cores are the high indices on MediaTek), so the
// order stays right even if cpufreq nodes are unreadable.
//
// Audio threads (ours at nice -16, AAudio's realtime callback) are never
// confined: on the leftover cores they compete with streaming and loading,
// and the little cores alone cannot decode XMA through a load spike.
#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <android/log.h>

#include <rex/cvar.h>

REXCVAR_DEFINE_UINT32(gta4_thread_pinning, 4, "GTA IV/Performance",
                      "Give the N busiest threads a big core each and keep the others off them "
                      "(0 = leave scheduling to Android, default 4 = Dimensity 8500 big cores)")
    .range(0, 4);

namespace {

struct ThreadStat {
  uint64_t ticks = 0;
  bool audio = false;  // high priority or realtime: left on every core
};

ThreadStat ReadThreadStat(pid_t tid) {
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
  FILE* file = std::fopen(path, "r");
  if (!file) return {};
  char buffer[512];
  const size_t length = std::fread(buffer, 1, sizeof(buffer) - 1, file);
  std::fclose(file);
  buffer[length] = 0;
  // Fields after the parenthesized name: state is field 3, utime 14, stime 15.
  const char* p = std::strrchr(buffer, ')');
  if (!p) return {};
  // Field 19 is nice, 41 the scheduling policy (1 FIFO, 2 RR).
  unsigned long long utime = 0, stime = 0;
  long nice = 0, policy = 0;
  int field = 2;
  for (const char* c = p + 1; *c && field < 41; ++c) {
    if (*c == ' ') {
      ++field;
      if (field == 14) utime = std::strtoull(c + 1, nullptr, 10);
      if (field == 15) stime = std::strtoull(c + 1, nullptr, 10);
      if (field == 19) nice = std::strtol(c + 1, nullptr, 10);
      if (field == 41) policy = std::strtol(c + 1, nullptr, 10);
    }
  }
  return {utime + stime, nice <= -16 ||policy == 1 || policy == 2};
}

long ReadCpuLong(int cpu, const char* leaf) {
  char path[112];
  std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/%s", cpu, leaf);
  long value = 0;
  if (FILE* file = std::fopen(path, "r")) {
    if (std::fscanf(file, "%ld", &value) != 1) value = 0;
    std::fclose(file);
  }
  return value;
}

// Fastest first: max frequency, then cpu_capacity, then the higher cpu index.
std::vector<int> CoresByMaxFrequency() {
  struct Core {
    long frequency, capacity;
    int cpu;
  };
  std::vector<Core> cores;
  const long count = sysconf(_SC_NPROCESSORS_CONF);
  for (int cpu = 0; cpu < count && cpu < CPU_SETSIZE; ++cpu) {
    cores.push_back({ReadCpuLong(cpu, "cpufreq/cpuinfo_max_freq"),
                     ReadCpuLong(cpu, "cpu_capacity"), cpu});
  }
  std::sort(cores.begin(), cores.end(), [](const Core& a, const Core& b) {
    if (a.frequency != b.frequency) return a.frequency > b.frequency;
    if (a.capacity != b.capacity) return a.capacity > b.capacity;
    return a.cpu > b.cpu;
  });
  std::vector<int> result;
  for (const Core& core : cores) result.push_back(core.cpu);
  return result;
}

void SetAffinity(pid_t tid, const cpu_set_t& set) {
  syscall(__NR_sched_setaffinity, tid, sizeof(set), &set);
}

void PinningLoop() {
  const std::vector<int> cores = CoresByMaxFrequency();
  if (cores.size() < 6) return;
  std::unordered_map<pid_t, uint64_t> previous;
  std::unordered_set<pid_t> held;  // threads on a dedicated core in the last pass
  uint32_t applied = 0;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // Never take more than half of the cores: the rest must stay for the others.
    const uint32_t pinned = std::min<uint32_t>(
        {REXCVAR_GET(gta4_thread_pinning), 4u, uint32_t(cores.size() / 2)});
    std::vector<std::pair<uint64_t, pid_t>> usage;
    std::vector<pid_t> audio;
    std::unordered_map<pid_t, uint64_t> current;
    if (DIR* directory = opendir("/proc/self/task")) {
      while (dirent* entry = readdir(directory)) {
        const pid_t tid = pid_t(std::atoi(entry->d_name));
        if (tid <= 0) continue;
        const ThreadStat stat = ReadThreadStat(tid);
        const uint64_t ticks = stat.ticks;
        current[tid] = ticks;
        if (stat.audio) {
          audio.push_back(tid);
          continue;
        }
        const auto old = previous.find(tid);
        uint64_t delta = old != previous.end() && ticks >= old->second ? ticks - old->second : 0;
        // Hysteresis: a pinned thread keeps its core unless overtaken by 25%+,
        // so threads are not migrated back and forth between cores.
        if (held.count(tid)) delta += delta / 4;
        usage.emplace_back(delta, tid);
      }
      closedir(directory);
    }
    previous.swap(current);
    if (!pinned && !applied) continue;
    std::sort(usage.rbegin(), usage.rend());
    cpu_set_t rest;
    CPU_ZERO(&rest);
    for (size_t i = pinned; i < cores.size(); ++i) CPU_SET(cores[i], &rest);
    held.clear();
    for (size_t i = 0; i < usage.size(); ++i) {
      cpu_set_t set;
      CPU_ZERO(&set);
      if (!pinned) {
        for (int cpu : cores) CPU_SET(cpu, &set);
      } else if (i < pinned) {
        CPU_SET(cores[i], &set);
        held.insert(usage[i].second);
      } else {
        set = rest;
      }
      SetAffinity(usage[i].second, set);
    }
    cpu_set_t all;
    CPU_ZERO(&all);
    for (int cpu : cores) CPU_SET(cpu, &all);
    for (pid_t tid : audio) SetAffinity(tid, all);
    if (pinned != applied) {
      __android_log_print(ANDROID_LOG_WARN, "LibertyRecomp", "thread-pinning: %u threads on cores %d,%d,%d,%d",
                          pinned, cores[0], cores[1], cores[2], cores[3]);
    }
    applied = pinned;
  }
}

const bool g_pinning_started = [] {
  std::thread(PinningLoop).detach();
  return true;
}();

}  // namespace
