import zarr

z = zarr.open('data/metaquest_dataset.zarr', mode='r')
for demo in z['data']:
    obs = z['data'][demo]['obs']
    actions = z['data'][demo]['actions']
    print(f"{demo}: actions={actions.shape}, images={obs['agentview_image'].shape}")
    assert actions.shape[0] == obs['agentview_image'].shape[0]


# 输出根 group 的属性
print("Root attrs:", dict(z.attrs))

# 检查 'data' group 是否存在
if 'data' not in z:
    raise ValueError("数据集缺少 'data' group")

# 检查每个 demo 的配置
for demo in z['data']:
    demo_group = z['data'][demo]
    print(f"Demo: {demo}")
    print("  attrs:", dict(demo_group.attrs))
    # 检查关键字段
    if 'obs' not in demo_group or 'actions' not in demo_group:
        raise ValueError(f"{demo} 缺少 'obs' 或 'actions'")
    obs = demo_group['obs']
    actions = demo_group['actions']
    print(f"  actions shape: {actions.shape}")
    if 'agentview_image' not in obs:
        raise ValueError(f"{demo} 的 obs 缺少 'agentview_image'")
    print(f"  agentview_image shape: {obs['agentview_image'].shape}")
    # 检查长度是否一致
    assert actions.shape[0] == obs['agentview_image'].shape[0]