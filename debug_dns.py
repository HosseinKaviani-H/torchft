import socket
import os

addr = os.environ.get("MASTER_ADDR", socket.gethostname())
print("MASTER_ADDR:", addr)
print("getfqdn(addr):", socket.getfqdn(addr))
try:
    print("gethostbyname(getfqdn(addr)):", socket.gethostbyname(socket.getfqdn(addr)))
except Exception as e:
    print("gethostbyname FAILED:", e)
print("getfqdn():", socket.getfqdn())
try:
    print("gethostbyname(getfqdn()):", socket.gethostbyname(socket.getfqdn()))
except Exception as e:
    print("gethostbyname() FAILED:", e)
print("hostname:", socket.gethostname())
try:
    print("gethostbyname(hostname):", socket.gethostbyname(socket.gethostname()))
except Exception as e:
    print("gethostbyname(hostname) FAILED:", e)
