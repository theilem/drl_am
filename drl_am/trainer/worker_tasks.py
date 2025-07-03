import queue
from time import sleep


def step_process(env, step_queue, step_result_queue):
    state = None
    done = True
    obs = reward = truncated = info = None
    while True:
        command, state_or_action = step_queue.get()
        if command == "reset":
            state = state_or_action
            done = False
        elif command == "step":
            action = state_or_action
            if not done:
                obs, reward, done, truncated, info = env.step(state, action)
            step_result_queue.put((obs, reward, done, truncated, info))
        elif command == "stop":
            break
        else:
            raise ValueError(f"Invalid command: {command}")


def reset_process(env, reset_queue, stop_event):
    while not stop_event.is_set():
        obs, state = env.reset()
        while not stop_event.is_set():
            try:
                reset_queue.put((obs, state), timeout=0.1)
                break
            except queue.Full:
                sleep(1)
    while not reset_queue.empty():
        reset_queue.get()
