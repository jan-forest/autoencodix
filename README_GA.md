
# Instructions for the GA Workshop 


1. Set up VPN according to instructions sent over via email a few days ago. 

2. Open JupyterLab via the following link (if you already have a ZIH account, copy paste it into an incognito browser):
https://jupyterhub.hpc.tu-dresden.de/hub/spawn#/~(cluster~'capella~nodes~'1~ntasks~'1~cpuspertask~'14~mempercpu~'2048~gres~'gpu*3a1~runtime~'05*3a30*3a00~reservation~'p_scads_trainings_85~project~'p_scads_trainings) 

3. Open Terminal, located in the main area.

4. Type in the following commands:
```
ws_allocate -F horse AE-ws 10
ws_register -F horse $HOME
cd $HOME/horse/$USER-AE-ws/
git clone -b ga-workshop https://github.com/jan-forest/autoencodix
chmod +x autoencodix/install-kernel.sh
```

5. In the left sidebar, you will find the file browser. Here you are in your `$HOME` directory. 
Navigate to (click on) horse/autoencodix and open `install-kernel-script.ipynb`.

6. Run the install-kernel-script.ipynb notebook.

7. Follow the instructions indicated in the output.  

### Troubleshooting:
If, for any reason, you would like to respawn JupyterLab, click: File -> Hub Control Panel -> Stop My Server
To close your session. Then use the link above in 2. to respawn.


1. `Error: could not create workspace directory!`

If you are encountering this error, please follow the instructions indicated [here](https://doc.zih.tu-dresden.de/data_lifecycle/workspaces/?h=workspace#faq-and-troubleshooting). 


2. My JupyterLab session stopped!

Use the link above in 2. to respawn.

3. Failing to spawn using the link provided.

You are most likely automatically logged in using your own ZIH credentials and not the ones provided by the workshop instructors for the day. Open an incognito tab and copy paste the link provided in 2. Then use the login credentials provided in the email sent a few days back. 
