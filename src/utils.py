import time
import tracemalloc


def get_memory_and_execution_time_details(func, is_teacher):
    tracemalloc.start()
    start_time = time.time()
    func(teacher=is_teacher)
    exec_time = time.time() - start_time
    print("Model Evaluation Time: ")
    print(exec_time)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 3}KB; Peak was {peak / 10 ** 3}KB")
    tracemalloc.stop()

    return current, peak, exec_time
