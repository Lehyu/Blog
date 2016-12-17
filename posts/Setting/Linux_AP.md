## Solution
1. Download **create_ap**

{%highlight C++%}
git clone https://github.com/oblique/create_ap
or
yaourt -S create_ap
{%endhighlight%}

2. Install

{%highlight C++%}
cd create_ap
make install
{%endhighlight%}

3. Create AP
choose one

{%highlight C++%}
create_ap wlan0 eth0 MyAccessPoint  #no passpharse
create_ap wlan0 eth0 MyAccessPoint MyPassPhrase #WPA+WPA2 passphrase
create_ap -n wlan0 MyAccessPoint MyPassPhrase  #AP without Internet sharing
create_ap -m bridge wlan0 eth0 MyAccessPoint MyPassPhrase #Bridged Internet sharing
{%endhighlight%}
