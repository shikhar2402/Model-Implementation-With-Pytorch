import threading
import time

def thread_function(number):
    print(f"Thread {number} starting")
    start_time = time.time()
    time.sleep(number)
    end_time = time.time()
    print(f"Thread {number} finishing")
    print(f"Time taken: {end_time - start_time} seconds")

thread1 = threading.Thread(target=thread_function , args=[3])
thread2 = threading.Thread(target=thread_function , args=[3])
thread3 = threading.Thread(target=thread_function , args=[3])

start_time = time.time()
thread1.start()
thread2.start()
thread3.start()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")


 