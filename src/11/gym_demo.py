import gym


env = gym.make('CartPole-v0')

o = env.reset()

for _ in range(1000):
    env.render()

    c_pos, c_vel, p_ang, p_vel = o

    if p_ang > 0:
        a = 1
    else:
        a = 0

    o, r, d, i = env.step(a)
