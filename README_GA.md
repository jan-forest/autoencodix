
# Instructions for the GA Workshop 


### 1. Set up VPN 
See instructions sent over via email a few days ago:
1. Download VPN client [here](https://www.eduvpn.org/client-apps).
2. Launch client and log in with your guest account.
3. If you have any errors, find solutions [here](https://faq.tickets.tu-dresden.de/otrs/public.pl?Action=PublicFAQZoom;ItemID=1284).

### 2. Open JupyterLab
- use the following link: copy/paste it into an **incognito/private browser window**:
https://jupyterhub.hpc.tu-dresden.de/hub/spawn#/~(cluster~'capella~nodes~'1~ntasks~'1~cpuspertask~'14~mempercpu~'2048~gres~'gpu*3a1~runtime~'05*3a30*3a00~reservation~'p_scads_trainings_85~project~'p_scads_trainings)
- login with your guest account
- in some browsers you will get this message, click 'ok'.
  <img width="720" height="488" alt="image" src="https://github.com/user-attachments/assets/258c49ee-5cd5-4f48-9583-e8029d286049" />


### 3. Open Terminal, located in the main area.

### 4. Type or copy/paste the following commands:
```
ws_allocate -F horse AE-ws 10
ws_register -F horse $HOME
cd $HOME/horse/$USER-AE-ws/
git clone -b ga-workshop https://github.com/jan-forest/autoencodix
chmod +x autoencodix/install-kernel.sh
```

### 5. Navigate to autoencodix directory
In the left sidebar, you will find the file browser. Here you are in your `$HOME` directory. 
Navigate to (click on) horse/autoencodix and open `install-kernel-script.ipynb`.
<img width="1002" height="475" alt="image" src="https://github.com/user-attachments/assets/185e2801-8056-4f83-ad39-0649829bf726" />


### 6. Run the install-kernel-script.ipynb notebook.

### 7. Follow the instructions indicated in the output.  

## Troubleshooting:
If, for any reason, you would like to respawn JupyterLab, click: File -> Hub Control Panel -> Stop My Server
To close your session. Then use the link above in 2. to respawn.


### 1. `Error: could not create workspace directory!`

If you are encountering this error, please follow the instructions indicated [here](https://doc.zih.tu-dresden.de/data_lifecycle/workspaces/?h=workspace#faq-and-troubleshooting). 


### 2. My JupyterLab session stopped!

Use the link above in 2. to respawn.

### 3. Failing to spawn using the link provided.

You are most likely automatically logged in using your own ZIH credentials and not the ones provided by the workshop instructors for the day. Open an incognito tab and copy paste the link provided in 2. Then use the login credentials provided in the email sent a few days back. 
