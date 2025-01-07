import h5py

file_path = "/cephfs/shared/yangrujia/mimicgen/image_depth.hdf5"
with h5py.File(file_path, 'r') as f:
    f = f['data']
    demos = list(f.keys())  # 获取所有 demo 的键
    print(f"Found demos: {demos}")

    # 2. 遍历每个 demo 并查看 agentview_image 信息
    for demo_name in demos:
        print(f"\nProcessing demo: {demo_name}")
        
        # 访问当前 demo 下的 agentview_image 数据
        demo_group = f[demo_name]
        print(demo_group.keys())
        print(demo_group['tracks'][0])
        print("-------- OBS ---------")
        # 获取 'obs/agentview_image' 数据集
        if 'obs' in demo_group and 'agentview_image' in demo_group['obs']:
            agentview_image = demo_group['obs']['agentview_image']
            
            # 打印图像的形状、数据类型等信息
            print(f"Shape of agentview_image: {agentview_image.shape}")
            print(f"Data type of agentview_image: {agentview_image.dtype}")
            print(f"Number of time steps (T): {agentview_image.shape[0]}")
            print(f"Image dimensions (H x W): {agentview_image.shape[1]} x {agentview_image.shape[2]}")
            print(f"Number of channels (C): {agentview_image.shape[3]}")
        else:
            print("No 'agentview_image' found in this demo.")
        print("-------- END ---------")  