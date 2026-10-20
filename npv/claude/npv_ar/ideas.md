# 想法清单

状态：待试 / 进行中 / 已保留 / 已回退 / 待审阅

## 待试
- [ ] URL 形态：列表页正则（/list、/tag、?page=）作为独立特征纳入拟合
- [ ] URL 形态：用户主页正则（/u/、/user/、/space/）作为独立特征纳入拟合
- [ ] 把现有实验特征 url_depth、is_root、text_len_log 纳入拟合，看哪些有增量
- [ ] pc 位拆平：把 pc_bit26、pc_bit10 等纳入拟合，替代乘法规则
- [ ] UGC 子域名单（bbs、blog、tieba、zhidao 等）→ site_ugc_flag，再试 spr_sr × ugc 交互项
- [ ] 删除作用小的特征，看能否更简单而持平

## 已试

## 信号申请
（格式：字段 | 能救哪类对 | train 证据：错例条数 + 3 个示例 URL）
