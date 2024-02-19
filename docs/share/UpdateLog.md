# 更新日志
---
记录修改内容用。


### 02.19
---

目前存在问题：无法获取静态资源图片。

1.构建VuePress

**准备工作：**

安装 `nvm` 或 `nodejs(>=18.16.0)`

使用[pnpm工具](https://www.pnpm.cn/)代替npm，
在控制台输入并安装pnpm：

```shell
iwr https://get.pnpm.io/install.ps1 -useb | iex
```

**手动创建VuePress：**

自动创建可能会有一些问题，这里按照官方[快速上手](https://v2.vuepress.vuejs.org/zh/guide/getting-started.html)
选择手动创建，并不麻烦。

启动命令：pnpm docs:dev ，热重载模式启动，方便调试。

2.添加静态资源到文档的使用方法，例如文件存放在public下就可以用下列方式调用：
```vue
![VuePress](/icon/相声团.png)
```

![VuePress](/icon/相声团.png)

3.侧边栏的设置方法：为sidebar设置Value=heading时，会根据markdown的标题属性自动设置侧边栏标题。
当然也可以手动设置。下面是首页的一个示例，我想要首页不显示侧边栏，而其他的文档自动设置：

```js
export default {
  theme: defaultTheme({
    // 侧边栏对象
    // 不同子路径下的页面会使用不同的侧边栏
    sidebar: {
      '/share/': [
        {
          text: 'Share',
          children: ['/guide/introduction.md', '/guide/getting-started.md'],
        },
      ],
      '/': 'heading',
    },
  }),
}

```
