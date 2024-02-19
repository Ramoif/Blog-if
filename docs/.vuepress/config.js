import {viteBundler} from '@vuepress/bundler-vite'
import {defaultTheme} from '@vuepress/theme-default'
import {defineUserConfig} from 'vuepress'

export default defineUserConfig({
    base: '/Blog-if',
    bundler: viteBundler(),
    theme: defaultTheme(
        {
            // 全局顶栏
            navbar: [
                {text: "首页", link: "/",},
                {text: "更新日志", link: "/share/UpdateLog",},
                {text: "分享", link: "/share/U2Net",},
            ],
            // 不同子路径的侧边栏
            // sidebar:{
            //
            // },
            // logo图标, 静态资源存放路径为.vuepress/public
            logo: '/icon/相声团.png',
            // 仓库连接的URL
            repo: 'https://github.com/Ramoif'
        }
    ),
})

