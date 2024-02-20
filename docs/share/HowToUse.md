# 分享
<img :src="$withBase('/icon/hero.png')" alt="VuePress Logo">


# 部署文档
---

1.打包项目
---
使用`pnpm docs:build`打包vuepress项目, 在`docs/.vuepress/dist`会生成打包好的项目。

上传到服务器即可。

2.nginx发布
---
修改nginx配置文件下列内容，重启服务后即可访问。
```txt
server {
        listen 80;
        listen [::]:80;
        server_name fensecaib;
        
        location / {
            alias   /www/wwwblog/dist/;
            index  index.html index.htm;
        }
    }
}
```

3.gitHub page
---
因群友访问网络问题，待施工。
