from pythonosc import dispatcher, osc_server

def dbg(addr, *args):
    print(addr, *args)

disp = dispatcher.Dispatcher()
disp.set_default_handler(dbg)

srv = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 8000), disp)
print("Waiting on {}".format(srv.server_address))
srv.serve_forever()
