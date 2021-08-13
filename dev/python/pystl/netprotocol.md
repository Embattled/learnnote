# Internet Protocols and Support

* 网络高级API, 提供了对多种协议的支持和处理

## 

```py
def http_post(url,data_json):
    jdata = json.dumps(data_json)
    req = requests.post(url, jdata)

    # convert response to dict
    re= json.loads(req.text)
    return re['remaining']
```