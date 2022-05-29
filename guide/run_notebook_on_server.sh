# THIS IS A GUIDE ON HOW TO START A JUPYTER SERVER ON AN IMS SERVER
# AND USE THE SERVERS RESOURCES WHILE WORKING FROM YOUR LOCAL COMPUTER

# NOTE: THIS ISN'T AN EXECUTABLE!

# STARTING SERVER
# 1) Connect to the server via SSH
ssh user@phoenix.ims

# 2) Change directory to desired location where the server will be run.
#    This is useful if you have a project and want to easily import some
#    local code into the notebook
cd ~/path/to/project

# 3) Create a Python virtual environment and activate it, so that you
#    don't pollute the entire server with your weeb packages
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate

# 4) Run command to start jupyter server. This will now give you an address
#    such as "localhost:XXXX", where XXXX will be the port number used in
#    the next steps to make a connection to the server
jupyter notebook

# CONNECTING TO SERVER
# 1) Enable ssh tunneling. 8000 is the local port of your computer
#    8889 is the port of the server, so from the instructions before,
#    it's equal to XXXX.
ssh -L 8000:localhost:8889 phoenix.ims

# RUNNING STUFF IN BROWSER
# 1) When starting the jupyter server, it will print out an address in this format
#    http://localhost:XXXX/?token=7b9249a85cae4f85d84e3be487cb407c3729ebdbaa07c511.
#    
#    So all you have to do is change the XXXX with the local port (so 8000) and run
#    the URL in the browser.
# 
#    Done. Now you can run stuff on the IMS servers.