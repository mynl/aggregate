

import ftplib
ftp = ftplib.FTP()
host = 'www.mynl.com'

ftp.connect(host)
ftp.getwelcome()

ftp.login(user='mynl0com',passwd =)

ftp.retrlines('LIST')
# https://www.atlantic.net/cloud-hosting/how-to-ftp-uploads-python/

ftp.cwd('/web/trash')
ftp.pwd()

with open(r'C:\temp\aggregate\_build\singlehtml\documentation.html', 'rb') as f:
    ftp.storbinary('STOR documentation.html', f)

ftp.quit()
