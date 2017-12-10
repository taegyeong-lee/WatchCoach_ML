import threading

def prints():
    print("a")
    while True:
        a=1

def a():
    print("b")

print("hello world")


t1 = threading.Thread(target=prints)
t1.start()
t2 = threading.Thread(target=a)
t2.start()



