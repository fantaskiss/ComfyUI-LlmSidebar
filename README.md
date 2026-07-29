简单复用[ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)的llama.cpp构建一个聊天与反推工具。
需要先安装该节点。
可以直接在loadimage与previewimage节点上右击，点击Discribe with llm来生成描述。
<img width="1683" height="639" alt="image" src="https://github.com/user-attachments/assets/1cd5c3f9-dc75-4dfe-91cb-8d284fdb158e" />
生成描述：
<img width="1377" height="907" alt="image" src="https://github.com/user-attachments/assets/111ab293-a792-4310-b90e-270a6f4eaae2" />
<img width="1160" height="748" alt="image" src="https://github.com/user-attachments/assets/d1f6ed0a-b9e8-4d21-8389-b52e4b001504" />
注意，运行主工作流前，视情况释放显存，点击红色upload按钮。
<img width="775" height="866" alt="image" src="https://github.com/user-attachments/assets/112d6df5-9050-4070-9aab-c9cce26a5f3d" />


歪脖编写，若有错误请自行或者同样歪脖修改。感谢ds！

20260729: 

增加了本地读取文件的简单loop engine，可以使用llm wiki
<img width="753" height="603" alt="ab5eacea046968ef88c61e6f22889fdd" src="https://github.com/user-attachments/assets/5421a275-5e03-4cae-a326-f61b661e4ced" />
<img width="1019" height="936" alt="5dd4c080cf5907c88bb2cf2af06261ba" src="https://github.com/user-attachments/assets/2593b109-0303-4c4b-bc44-33c1c7d8b891" />
