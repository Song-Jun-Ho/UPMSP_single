from agent.network import *
from environment.New_Parallel_Machine import Forming
from environment.Steelplate import *


episode = 0    # total episode count
score_avg = 0
score_max = 0
num_episode = 50000


class A3Cagent():
    def __init__(self, action_size, state_dim):
        self.state_size = (state_dim,)
        self.state_dim = state_dim
        self.action_size = action_size

        # A3C Hyper-parameter
        self.discount_factor = 0.99
        self.lr = 5e-5
        self.threads = 8

        # Generating global network
        self.global_model = ActorCritic(self.action_size, self.state_size)
        # Initiating weights of global network
        self.global_model.build(tf.TensorShape((None, *self.state_size)))

        # Generating Optimizer function
        self.optimizer = AdamOptimizer(self.lr, use_locking=True)

        # Path to save trained global network model
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')


    def train(self):
        # Generating Runner class
        runners = []
        for i in range(self.threads):
            worker_name = 'worker_{0}'.format(i)
            df, index, process_all, process_list, m_type_dict, machine_num = import_steel_plate_schedule(
                '../environment/data/forming_data.csv')
            steel_plate = SteelPlate(df, index, process_list, m_type_dict)
            forming_shop = Forming(parts=steel_plate.parts, process_all=process_all, machine_num=machine_num)

            runners.append(Runner(forming_shop, worker_name, self.action_size, self.state_size, self.state_dim, self.global_model,
                                  self.optimizer, self.discount_factor, self.model_path))

        for i, runner in enumerate(runners):
            print("Start worker #{:d}".format(i))
            runner.start()



class Runner(threading.Thread):
    def __init__(self, env, name, action_size, state_size, state_dim, global_model, optimizer, discount_factor,
                 model_path):
        threading.Thread.__init__(self)

        self.name = name
        self.action_size = action_size
        self.state_size = state_size
        self.state_dim = state_dim
        self.global_model = global_model
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.model_path = model_path

        self.states = []
        self.actions = []
        self.rewards = []

        # Generating local network, environment, and tensorboard
        self.local_model = ActorCritic(self.action_size, self.state_size)
        self.env = env

        # Tensorboard configuration
        self.writer = tf.summary.create_file_writer('./summary/a3c/train/' + self.name)

        # variables to record
        self.avg_loss = 0
        self.avg_p_max = 0

        # n-time-step configuration
        self.t_max = 20
        self.t = 0


    def draw_tensorboard(self, score, mean_weighted_tardiness, make_span, step, v_l, p_l, e_l, e):
        avg_p_max = self.avg_p_max / float(step)
        with self.writer.as_default():
            tf.summary.scalar("Perf/Total Reward", score, step=e)
            tf.summary.scalar("Perf/Mean Weighted Tardiness", mean_weighted_tardiness, step=e)
            tf.summary.scalar("Perf/Make Span", make_span, step=e)
            tf.summary.scalar("Average Max Prob", avg_p_max, step=e)
            tf.summary.scalar("Perf/Duration", step, step=e)
            tf.summary.scalar("Loss/Value loss", v_l, step=e)
            tf.summary.scalar("Loss/Policy loss", p_l, step=e)
            tf.summary.scalar("Loss/Entropy", e_l, step=e)



    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        policy = self.local_model(state)[0][0]
        policy = tf.nn.softmax(policy)
        #print(policy)
        action = np.random.choice(self.action_size, 1, p=policy.numpy())[0]
        return action, policy


    def append_sample(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        act = np.reshape(act, [1, self.action_size])
        self.actions.append(act)
        self.rewards.append(reward)

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)
        return unpack

    # n-step time difference targets
    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value[0]

        for k in reversed(range(0, len(rewards))):
            cumulative = self.discount_factor * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets


    def compute_loss(self, done, next_v_value):
        n_step_td_targets = self.n_step_td_target(self.rewards, next_v_value, done)
        n_step_td_targets = tf.convert_to_tensor(n_step_td_targets, dtype=tf.float32)

        states = np.reshape(self.states, [-1, self.state_dim])
        policy, values = self.local_model(states)

        # critic loss
        advantages = n_step_td_targets - values
        critic_loss = 0.5 * tf.reduce_sum(tf.square(advantages))

        # actor loss
        action = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        policy_prob = tf.nn.softmax(policy)
        action_prob = tf.reduce_sum(policy_prob * action, axis=1, keepdims=True)
        cross_entropy = -tf.math.log(action_prob + 1e-10)
        actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))

        entropy = tf.reduce_sum(policy_prob*tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        actor_loss_with_entropy = actor_loss + 0.01*entropy

        total_loss = 0.5 * critic_loss + actor_loss_with_entropy

        return total_loss, actor_loss, critic_loss, entropy

    # calculate gradients from local network and update global network with the calculated gradients
    def train_model(self, done, next_v_value):
        global_params = self.global_model.trainable_variables
        local_params = self.local_model.trainable_variables

        with tf.GradientTape() as tape:
            total_loss, actor_loss, critic_loss, entropy = self.compute_loss(done, next_v_value)

        grads = tape.gradient(total_loss, local_params)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)

        self.optimizer.apply_gradients(zip(grads, global_params))

        self.local_model.set_weights(self.global_model.get_weights())

        self.states, self.actions, self.rewards = [], [], []

        return actor_loss, critic_loss, entropy

    def run(self):
        # global variables shared by actor-runners
        global episode, score_avg, score_max

        while episode < num_episode:
            episode_reward, step, done = 0, 0, False

            part_sent = []
            action_list = []

            state = self.env.reset()

            while not done:
                state = np.reshape(state, [1, self.state_dim])
                action, policy = self.get_action(state)
                next_state, reward, done = self.env.step(action)

                reward = np.reshape(reward, [1, 1])

                self.append_sample(state, action, reward)

                state = next_state
                step += 1
                self.t += 1
                episode_reward += reward[0][0]

                part_sent.append(self.env.model['LH'].parts_routed)
                action_list.append(action)
                
                # 정책확률의 최댓값
                self.avg_p_max += np.amax(policy.numpy())

                # if episode is done or time step reaches max time step, start training
                if self.t >= self.t_max or done:
                    # Extract samples from batch
                    self.states = self.unpack_batch(self.states)
                    self.actions = self.unpack_batch(self.actions)
                    self.rewards = self.unpack_batch(self.rewards)

                    next_state = np.reshape(next_state, [1, self.state_dim])
                    _, next_v_value = self.local_model(next_state)

                    actor_loss, critic_loss, entropy = self.train_model(done, next_v_value)
                    #print('update')

                    self.t = 0

                    if done:
                        episode += 1
                        score_max = episode_reward if episode_reward > score_max else score_max
                        score_avg = 0.9*score_avg + 0.1*episode_reward if score_avg!=0 else episode_reward

                        log = "episode: {:5d} | reward: {:4.1f} | ".format(episode, episode_reward)
                        #log += "score max: {:4.1f} | ".format(score_max)
                        #log += "score avg: {:.3f} | ".format(score_avg)
                        log += "episode length: {0} | ".format(step)
                        log += "value loss: {:5.4f} | policy loss: {:5.4f} | entropy: {:5.4f} | ".format(critic_loss,
                                                                                                         actor_loss, entropy)
                        print(log)
                        print("parts sent: " + str(part_sent))
                        print("action in each step: " + str(action_list))
                        mean_weighted_tardiness = self.env.mean_weighted_tardiness
                        make_span = self.env.make_span

                        self.draw_tensorboard(episode_reward, mean_weighted_tardiness, make_span, step, critic_loss,
                                              actor_loss, entropy, episode)

                        self.avg_p_max = 0

                        if episode % 250 == 0 and self.name == 'worker_0':
                            self.global_model.save_weights(self.model_path, save_format="tf")



if __name__ == "__main__":
    global_agent = A3Cagent(action_size=4, state_dim=64)
    global_agent.train()


