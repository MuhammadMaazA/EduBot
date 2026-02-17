import asyncio
import websockets
import threading
from threading import Thread
import time
import functools
from http.server import BaseHTTPRequestHandler, HTTPServer

lock = threading.Lock()
global_data = None
global_data_received = False
t = None

clients = set()

# Handler for the server
async def handler(websocket, path, callback):
    clients.add(websocket)  # Register new client
    try:
        async for message in websocket:
            print(f"Received message from {websocket.remote_address}: {message}")
            # Check message type
            if isinstance(message, str):
                print(f"[server.py]: Received user input from the web browser: {str(message)}")
                data = str(message)
                if data == "close":  # If the client sends "close", close the connection and exit the program
                    callback(data)
                    await websocket.close()
                    exit(0)
                callback(data)  # Execute the callback function with the received data
    finally:
        clients.remove(websocket)

# Run the server
def run_server(callback):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(functools.partial(handler, callback=callback), "localhost", 10000)
    loop.run_until_complete(start_server)
    loop.run_forever()

async def send_data(feedback):
    if clients:
        print(f"[server.py>send_data]: Received user input from state machine")
        await asyncio.gather(*(client.send(feedback) for client in clients))
    else:
        print("no connected clients")

def main_callback(data):
    global lock
    global global_data
    global global_data_received

    print("before lock")
    lock.acquire()

    print("after lock")
    
    global_data_received = True
    global_data = data
    lock.release()

# Wait for a response from the client (web module)
def await_response():
    global lock
    global global_data
    global global_data_received
    while True:
        lock.acquire()
        if global_data_received:
            global_data_received = False
            temp = global_data
            lock.release()
            return temp
        lock.release()
        time.sleep(0.1)

# Start a new thread for the server
def start_thread():
    global t
    t = Thread(target=run_server, args=(main_callback,))
    t.start()

def join_thread():
    global t
    t.join()

if __name__ == "__main__":
    print("thread started")
    start_thread()
