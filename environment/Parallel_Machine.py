import simpy
import pandas as pd
import random
import os
import functools

import numpy as np

from environment.RL_SimComponent import *

# weight = [1, 1.2, 1.4, 1.5, 1.6, 2, 2.1, 2.3, 2.5, 2.6]
# job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# machine_num = {'LH': 8}
# expon_dist = functools.partial(np.random.exponential, size=None)
# # job type 별 average process time
# process_time = [30, 40, 55, 60, 80, 70, 65, 83, 66, 92]
# process_list = ['LH', 'Sink']
# process_all = ['LH']
# priority = {'LH': [1, 1, 1, 2, 2, 2, 2, 2]}
# IAT = 50


class Forming(object):
    def __init__(self, weight, job_type, machine_num, process_time, process_list, process_all, priority, IAT, part_num, K):
        self.weight = weight
        self.job_type = job_type
        self.process_time = process_time
        self.process_list = process_list
        self.process_all = process_all
        self.machine_num = machine_num
        self.priority = priority
        self.IAT = IAT
        self.part_num = part_num
        self.K = K

        self.env, self.model, self.source = self._modeling()
        self.done = False

        self.mean_weighted_tardiness = 0
        self.make_span = 0


    def _modeling(self):
        # Make model
        env = simpy.Environment()
        model = dict()

        source = Source(env, self.IAT, self.weight, self.job_type, self.process_time, model, self.process_list, self.machine_num['LH'], self.part_num, self.K)

        for i in range(len(self.process_all) + 1):
            if i == len(self.process_all):
                model['Sink'] = Sink(env)
            else:
                model[self.process_all[i]] = Process(env, self.process_all[i], self.machine_num[self.process_all[i]],
                                                self.priority[self.process_all[i]],
                                                model, self.process_list)

        return env, model, source


    def step(self, action):
        ### 기본적으로 의사결정을 내리는 시점은 idle machine이 존재하고 동시에 buffer에 Part가 있을 때이다.
        self.done = False        # buffer_to_machine.items != 0 and machine_store.items != 0 인 상태에서 take action
        self.model['LH'].action = action
        current_time_step = self.env.now
        self.env.process(self.model['LH'].to_machine())

        while True:
            #print(self.env.now, "  Hi ", self.model['LH'].parts_routed)
            #print("Hi ", len(self.model['LH'].buffer_to_machine.items))
            buffer_to_machine = len(self.model['LH'].buffer_to_machine.items)
            #print("Hi ", len(self.model['LH'].machine_store.items))
            self.env.step()     # env.step()으로 event by event로 진행
            if len(self.model['LH'].buffer_to_machine.items) == 0 or len(self.model['LH'].machine_store.items) == 0:
                # if routing finished
                while self.env.peek() == self.env.now:
                    # env.now (현재 시점) 에 일어나는 모든 next scheduled event를 진행
                    self.env.run(self.env.timeout(0))
                    #print("Hi ", self.model['LH'].parts_routed)
                break

        #print('parts routed : ', self.model['LH'].parts_routed)

        if self.model['LH'].parts_routed == self.part_num:
            self.done = True

        if self.done:
            self.env.run()
        else:
            while True:
                if len(self.model['LH'].buffer_to_machine.items) != 0 and len(self.model['LH'].machine_store.items) != 0:
                    # simulation runs until the next decision step
                    break

                self.env.step()

        next_time_step = self.env.now

        next_state = self._get_state()

        reward = self._calculate_reward(current_time_step, next_time_step)


        return next_state, reward, self.done


    def _get_state(self):
        # f_1 (feature 1)
        # f_2 (feature 2)
        # f_3 (feature 3)
        # f_4 (feature 4)
        f_1 = np.zeros(len(self.job_type))
        NJ = np.zeros(len(self.job_type))

        f_2 = np.zeros(len(self.model['LH'].machines))

        z = np.zeros(len(self.model['LH'].machines))
        f_3 = np.zeros(len(self.model['LH'].machines))

        f_4 = np.zeros(len(self.model['LH'].machines))
        # machine_list 에서,  machine 0, machine 1 : auto,  machine 2 ~ machine 7 : Manu
        for i, machine in enumerate(self.model['LH'].machines):
            if machine not in self.model['LH'].machine_store.items:   # if the machine is not idle(working)
                step = machine.machine.items[0].step

                # feature 2
                f_2[i] = machine.machine.items[0].type / len(self.job_type)

                # feature 4
                f_4[i] = (machine.machine.items[0].due_date - self.env.now) / machine.machine.items[0].process_time
                # z_i : remaining process time of part in machine i
                if machine.machine.items[0].real_proc_time > self.env.now - machine.start_work:
                    z[i] = machine.machine.items[0].real_proc_time - (self.env.now - machine.start_work)
                    # feature 3
                    f_3[i] = z[i] / machine.machine.items[0].process_time

        # features to represent the tightness of due date allowance of the waiting jobs
        # f_5 (feature 5)
        # f_6 (feature 6)
        # f_7 (feature 7)
        # f_8 (feature 8)
        # f_9 (feature 9)
        f = [[] for _ in range(len(self.job_type))]
        f_5 = np.zeros(len(self.job_type))
        f_6 = np.zeros(len(self.job_type))
        f_7 = np.zeros(len(self.job_type))

        f_8 = np.zeros((len(self.job_type), 4))
        #f_9 = np.zeros((len(self.job_type), 4))

        if len(self.model['LH'].buffer_to_machine.items) == 0:
            f_5 = np.zeros(len(self.job_type))
            f_6 = np.zeros(len(self.job_type))
            f_7 = np.zeros(len(self.job_type))

            f_8 = f_8.flatten()
            #f_9 = np.zeros(len(self.job_type)*4)

        else:
            # interval number indicating the tightness of the due date allowance
            g = np.zeros((len(self.job_type), 4))
            # interval number indicating the process time
            h = np.zeros((len(self.job_type), 4))

            for i, part in enumerate(self.model['LH'].buffer_to_machine.items):

                NJ[part.type] += 1
                f[part.type].append((part.due_date - self.env.now) / part.process_time)
                # print('hi ', part.due_date - self.env.now)
                # print(part.due_date)
                # print(self.env.now)
                # print('ho ',part.process_time)

                # case for interval number g
                if (part.due_date - self.env.now) >= part.max_process_time:
                    g[part.type][0] += 1
                elif (part.due_date - self.env.now) >= part.min_process_time \
                        and (part.due_date - self.env.now) < part.max_process_time:
                    g[part.type][1] += 1
                elif (part.due_date - self.env.now) >= 0 and (part.due_date - self.env.now) < part.min_process_time:
                    g[part.type][2] += 1
                elif (part.due_date - self.env.now) < 0:
                    g[part.type][3] += 1

            # feature 1
            f_1 = np.array([2**(-1/nj) if nj > 0 else 0 for nj in NJ])

            # feature 8
            for j in self.job_type:
                for _g in range(4):
                    f_8[j][_g] = 2**(-1/g[j][_g]) if g[j][_g] != 0 else 0
            f_8 = f_8.flatten()


            for i in range(len(self.job_type)):
                if len(f[i]) != 0:
                    min_tightness = np.min(np.array(f[i]))
                    max_tightness = np.max(np.array(f[i]))
                    avg_tightness = np.average(np.array(f[i]))
                else:
                    min_tightness = 0
                    max_tightness = 0
                    avg_tightness = 0

                # feature 5
                f_5[i] = min_tightness
                # feature 6
                f_6[i] = max_tightness
                # feature 7
                f_7[i] = avg_tightness



            # for i, part in enumerate(self.model['LH'].buffer_to_machine.items):
            #     if (part.avg_proc_time[part.step] >= avg_of_avg_proc_time + 2*std_of_avg_proc_time) and (part.avg_proc_time[part.step]
            #                                                                                              < max_of_avg_proc_time):
            #         h[5] += 1
            #     elif (part.avg_proc_time[part.step] >= avg_of_avg_proc_time + std_of_avg_proc_time) and (part.avg_proc_time[part.step]
            #     < avg_of_avg_proc_time + 2*std_of_avg_proc_time):
            #         h[4] += 1
            #     elif (part.avg_proc_time[part.step] >= avg_of_avg_proc_time) and (part.avg_proc_time[part.step]
            #     < avg_of_avg_proc_time + std_of_avg_proc_time):
            #         h[3] += 1
            #     elif (part.avg_proc_time[part.step] >= avg_of_avg_proc_time - std_of_avg_proc_time) and (part.avg_proc_time[part.step]
            #     < avg_of_avg_proc_time):
            #         h[2] += 1
            #     elif (part.avg_proc_time[part.step] >= avg_of_avg_proc_time - 2*std_of_avg_proc_time) and (part.avg_proc_time[part.step]
            #     < avg_of_avg_proc_time - std_of_avg_proc_time):
            #         h[1] += 1
            #     elif (part.avg_proc_time[part.step] >= min_of_avg_proc_time) and (part.avg_proc_time[part.step] < avg_of_avg_proc_time
            #                                                                       - 2*std_of_avg_proc_time):
            #         h[0] += 1

        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8), axis=None)

        # Calculating mean-weighted-tardiness
        # Calculating make span
        if len(self.model['Sink'].sink) != 0:
            mean_w_tardiness = 0
            make_span = self.model['Sink'].sink[0].completion_time
            for part in self.model['Sink'].sink:
                mean_w_tardiness += part.weight * max(0, part.completion_time - part.due_date)
                if part.completion_time > make_span:
                    make_span = part.completion_time

            self.mean_weighted_tardiness = mean_w_tardiness / len(self.model['Sink'].sink)
            self.make_span = make_span

        return state


    def reset(self):
        self.env, self.model, self.source = self._modeling()
        self.done = False
        
        ######### 첫 decision step까지 시뮬레이션 이동 ##########
        while True:
            self.env.step()
            if len(self.model['LH'].buffer_to_machine.items) != 0 and len(self.model['LH'].machine_store.items) != 0:
                # simulation runs until the next decision step
                break

        return self._get_state()

    def _calculate_reward(self, current_time_step, next_time_step):
        # calculate reward for parts in waiting queue
        sum_reward_for_tardiness = 0
        sum_reward_for_makespan = 0
        for part in self.model['LH'].buffer_to_machine.items:
            if part.due_date < current_time_step:
                sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - current_time_step)
            elif part.due_date >= current_time_step and part.due_date < next_time_step:
                sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - part.due_date)

        for i, machine in enumerate(self.model['LH'].machines):
            if machine not in self.model['LH'].machine_store.items:
                if machine.machine.items[0].due_date < current_time_step:
                    sum_reward_for_tardiness += machine.machine.items[0].weight * (-2)*(next_time_step - current_time_step)
                elif machine.machine.items[0].due_date >=current_time_step and machine.machine.items[0].due_date < next_time_step:
                    sum_reward_for_tardiness += machine.machine.items[0].weight * (-2)*(next_time_step - machine.machine.items[0].due_date)

        sum_reward_for_tardiness = sum_reward_for_tardiness / 100

        if self.done == True:
            max_completion_time = self.model['Sink'].sink[0].completion_time
            for part in self.model['Sink'].sink:
                if part.completion_time > max_completion_time:
                    max_completion_time = part.completion_time
            sum_reward_for_makespan = 1 / max_completion_time
        
        # 선형적으로 더할 시 각각에 대한 coefficient가 필요
        sum_reward = sum_reward_for_makespan + sum_reward_for_tardiness

        # sum_reward_for_tardiness += 1

        return sum_reward_for_tardiness


if __name__ == '__main__':

    job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weight = np.random.uniform(0, 5, len(job_type))
    machine_num = {'LH': 8}
    # job type 별 average process time
    # process_time = [5, 9, 12, 16, 18, 23, 25, 28, 30, 32]
    p_ij = np.random.uniform(1, 20, size=(len(job_type), machine_num['LH']))
    p_j = np.average(p_ij, axis=1)
    process_list = ['LH', 'Sink']
    process_all = ['LH']
    priority = {'LH': [1, 2, 3, 4, 5, 6, 7, 8]}
    # IAT = 5
    arrival_rate = machine_num['LH'] / np.average(p_j)
    IAT = 1 / arrival_rate
    part_num = 300
    # due date generating factor
    K = 1

    forming_shop = Forming(weight=weight, job_type=job_type, machine_num=machine_num, process_time=p_ij,
                           process_list=process_list, process_all=process_all, priority=priority, IAT=IAT, part_num=part_num, K=K)

    #print(len(forming_shop.model['LH'].buffer_to_machine.items))
    #print(len(forming_shop.model['LH'].machine_store.items))
    for i in range(500):
        next_state1, reward1, done1 = forming_shop.step(0)

        print(len(forming_shop.model['LH'].buffer_to_machine.items))
        print(len(forming_shop.model['LH'].machine_store.items))
        print(len(forming_shop.model['Sink'].sink))
        print(next_state1)
        print(reward1)
        print(done1)

    print(forming_shop.mean_weighted_tardiness)

    #print(forming_shop.model['Sink'].sink[0].completion_time)
    # LH가 직전 공정인 BH보다 capacity가 크기 때문에 LH가 병목공정이 되고
    # buffer to machine 에는 part가 많이 쌓이는 반면 idle machine은 하나씩 발생하여
    # idle machine 하나가 발생하면 어떤 part를 먼저 해당 idle machine에 투입할 지 agent가 결정해준다.
