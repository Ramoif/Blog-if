import {viteBundler} from '@vuepress/bundler-vite'
import {defaultTheme} from '@vuepress/theme-default'
import {defineUserConfig} from 'vuepress'

export default defineUserConfig({
    // 这里会改变路由路径，例如本地部署会变为localhost:8080/Blog-if/
    base: '/Blog-if',
    title: '粉色彩笔空间',
    description:'Blog-if : 某个彩笔的文档网站',
    // 打包工具, vite
    bundler: viteBundler(),
    // 配置主题
    theme: defaultTheme(
        {
            // 全局顶栏
            navbar: [
                {text: "首页", link: "/",},
                {text: "更新日志", link: "/share/UpdateLog",},
                {text: "分享", link: "/share/U2Net",},
            ],
            // 不同子路径的侧边栏
            sidebar: {
                '/share/': [{
                    text: '分享',
                    children: ['/share/UpdateLog.md', '/share/HowToUse.md', '/share/U2Net.md']
                }],
            },
            // logo图标, 静态资源存放路径为.vuepress/public
            logo: '/icon/相声团.png',
            // 仓库连接的URL
            // 编辑示例：https://github.com/Ramoif/Blog-if/edit/main/docs/README.md
            editLinkPattern: ':repo/edit/:branch/docs/:path',
            repo: 'Ramoif/Blog-if',
            docsRepo: 'Ramoif/Blog-if',
            docsBranch: 'main',
        }
    ),
})

