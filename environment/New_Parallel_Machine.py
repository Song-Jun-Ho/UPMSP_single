import simpy
import pandas as pd
import random
import os
import functools

import numpy as np

from environment.RL_SimComponent import *
from environment.Steelplate import *


class Forming(object):
    def __init__(self, parts, process_all, machine_num):
        self.parts = parts
        self.process_all = process_all
        self.machine_num = machine_num
        self.env, self.model, self.source, self.monitor = self._modeling('../result/event_log_master_plan.csv')
        self.done = False
        self.mean_weighted_tardiness = 0
        self.make_span = 0

    def _modeling(self, event_path):
        # Make model
        env = simpy.Environment()
        model = dict()
        monitor = Monitor(event_path)

        # MTBF and MTTR distribution
        MTBF = functools.partial(np.random.triangular, left=9 * 24 * 60, mode=10 * 24 * 60, right=11 * 24 * 60)
        MTTR = functools.partial(np.random.triangular, left=3.6 * 60, mode=4 * 60, right=4.4 * 60)

        source = Source(env, self.parts, model, monitor)

        for i in range(len(self.process_all) + 1):
            if i == len(self.process_all):
                model['Sink'] = Sink(env, monitor)
            else:
                model[self.process_all[i]] = Process(env, self.process_all[i], self.machine_num[self.process_all[i]],
                                                     model, monitor, self.parts, MTTR=MTTR, MTBF=MTBF)

        return env, model, source, monitor

    def step(self, action):
        ### 기본적으로 의사결정을 내리는 시점은 idle machine이 존재하고 동시에 buffer에 Part가 있을 때이다.
        self.done = False  # buffer_to_machine.items != 0 and machine_store.items != 0 인 상태에서 take action
        self.model['LH'].action = action
        current_time_step = self.env.now
        self.env.process(self.model['LH'].to_machine())

        while True:
            # print(self.env.now, "  Hi ", self.model['LH'].parts_routed)
            # print("Hi ", len(self.model['LH'].buffer_to_machine.items))
            buffer_to_machine = len(self.model['LH'].buffer_to_machine.items)
            # print("Hi ", len(self.model['LH'].machine_store.items))
            self.env.step()  # env.step()으로 event by event로 진행
            if len(self.model['LH'].buffer_to_machine.items) == 0 or len(self.model['LH'].machine_store.items) == 0:
                # if routing finished
                while self.env.peek() == self.env.now:
                    # env.now (현재 시점) 에 일어나는 모든 nexted scheduled event를 진행
                    self.env.run(self.env.timeout(0))
                    # print("Hi ", self.model['LH'].parts_routed)
                break

        # print('parts routed : ', self.model['LH'].parts_routed)

        if self.model['LH'].parts_routed == 1000:
            self.done = True

        if self.done:
            self.env.run()
        else:
            while True:
                if len(self.model['LH'].buffer_to_machine.items) != 0 and len(
                        self.model['LH'].machine_store.items) != 0:
                    # simulation runs until the next decision step
                    break

                self.env.step()
                '''
                print('really  ', len(self.model['LH'].buffer_to_machine.items))
                print('really  ', len(self.model['LH'].machine_store.items))
                print(self.model['LH'].parts_routed)
                for key in self.process_all:
                    print(key + '  ' + str(self.model[key].parts_sent))
                    print(key + '  ' + 'buffer to machine : ')
                    print(self.model[key].buffer_to_machine.items)
                    print(key + '  ' + 'buffer to process : ')
                    print(self.model[key].buffer_to_process.items)
                print('Sink' + '   ' + 'part arrived : ' + str(self.model['Sink'].parts_rec))
                print('Length parts : ', len(self.parts))       '''

        next_time_step = self.env.now

        next_state = self._get_state()

        reward = self._calculate_reward(current_time_step, next_time_step)

        return next_state, reward, self.done

    def _get_state(self):
        # f_1 (feature 1)
        # f_2 (feature 2)
        # f_3 (feature 3)
        f_1 = np.zeros(len(self.model['LH'].machine_list))
        z = np.zeros(len(self.model['LH'].machine_list))
        f_2 = np.zeros(len(self.model['LH'].machine_list))
        f_3 = np.zeros(len(self.model['LH'].machine_list))
        # machine_list 에서,  machine 0, machine 1 : auto,  machine 2 ~ machine 7 : Manu
        for i, machine in enumerate(self.model['LH'].machine_list):
            if machine not in self.model['LH'].machine_store.items:  # if the machine is not idle(working)
                step = machine.machine.items[0].step
                f_1[i] = 1
                # print(machine.machine.items[0])
                # print(machine.machine.items[0].avg_proc_time)
                f_3[i] = (machine.machine.items[0].due_date - self.env.now) / machine.machine.items[0].avg_proc_time[
                    step]
                # z_i : remaining process time of part in machine i
                if machine.machine.items[0].real_proc_time > self.env.now - machine.start_work:
                    z[i] = machine.machine.items[0].real_proc_time - (self.env.now - machine.start_work)
                    f_2[i] = z[i] / machine.machine.items[0].avg_proc_time[step]

        # features to represent the tightness of due date allowance of the waiting jobs
        # f_4 (feature 4) -> 상위 20개 part에 대해서 process time
        # f_5 (feature 5) -> 상위 20개 part에 대해서 due date까지 tightness 정보

        f_4 = np.zeros(20)
        f_5 = np.zeros(20)

        if len(self.model['LH'].buffer_to_machine.items) != 0:
            mean_proc_time = 0
            for j in self.model['LH'].buffer_to_machine.items:
                mean_proc_time += j.avg_proc_time[j.step]
                W_j = list()
                for i in self.model['LH'].machine_store.items:
                    p_ij = j.process_time_dict[j.step][i.name]
                    w_ij = p_ij / j.weight
                    W_j.append(w_ij)
                j.W_ij = min(W_j)
            self.model['LH'].buffer_to_machine.items.sort(key=lambda part: part.W_ij)

            mean_proc_time = mean_proc_time / len(self.model['LH'].buffer_to_machine.items)

            for idx, part in enumerate(self.model['LH'].buffer_to_machine.items):
                if idx < 20:
                    f_4[idx] = part.avg_proc_time[part.step] / mean_proc_time
                    f_5[idx] = (part.due_date - self.env.now) / mean_proc_time
                else:
                    break


        state = np.concatenate((f_1, f_2, f_3, f_4, f_5), axis=None)    # state dim = 64

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
        df, index, process_all, process_list, m_type_dict, machine_num = import_steel_plate_schedule(
            '../environment/data/forming_data.csv')
        self.parts = SteelPlate(df, index, process_list, m_type_dict).parts
        self.env, self.model, self.source, self.monitor = self._modeling('../result/event_log_master_plan.csv')
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
                sum_reward_for_tardiness += part.weight * 1 * (part.due_date - current_time_step)
            elif part.due_date >= next_time_step:
                sum_reward_for_tardiness += part.weight * 1 * (next_time_step - current_time_step)

        sum_reward_for_tardiness = sum_reward_for_tardiness / 1000 + 10

        if self.done == True:
            max_completion_time = self.model['Sink'].sink[0].completion_time
            for part in self.model['Sink'].sink:
                if part.completion_time > max_completion_time:
                    max_completion_time = part.completion_time
            sum_reward_for_makespan = 10000000 / max_completion_time

        # 선형적으로 더할 시 각각에 대한 coefficient가 필요
        sum_reward = sum_reward_for_makespan + sum_reward_for_tardiness

        return sum_reward_for_tardiness


if __name__ == '__main__':
    df, index, process_all, process_list, m_type_dict, machine_num = import_steel_plate_schedule(
        './data/forming_data.csv')
    steel_plate = SteelPlate(df, index, process_list, m_type_dict)
    print(steel_plate.parts)

    forming_shop = Forming(parts=steel_plate.parts, process_all=process_all, machine_num=machine_num)
    print(len(forming_shop.parts))

    # print(len(forming_shop.model['LH'].buffer_to_machine.items))
    # print(len(forming_shop.model['LH'].machine_store.items))

    next_state1, reward1, done1 = forming_shop.step(0)

    print(len(forming_shop.model['LH'].buffer_to_machine.items))
    print(len(forming_shop.model['LH'].machine_store.items))
    print(next_state1)
    print(reward1)
    print(done1)

    next_state2, reward2, done2 = forming_shop.step(0)
    print(len(forming_shop.model['LH'].buffer_to_machine.items))
    print(len(forming_shop.model['LH'].machine_store.items))
    print(next_state2)
    print(reward2)
    print(done2)

    # print(forming_shop.model['Sink'].sink[0].completion_time)
    # LH가 직전 공정인 BH보다 capacity가 크기 때문에 LH가 병목공정이 되고
    # buffer to machine 에는 part가 많이 쌓이는 반면 idle machine은 하나씩 발생하여
    # idle machine 하나가 발생하면 어떤 part를 먼저 해당 idle machine에 투입할 지 agent가 결정해준다.