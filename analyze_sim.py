import torch
import os

data_dir = 'data/encodings'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')], reverse=True)
m_norms = []
i_norms = []
file_list = []

for f in files:
    try:
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        m_emb = data.get('memory_state')
        i_emb = data.get('input_tensor')
        if m_emb is not None and i_emb is not None:
            mv = m_emb.flatten()[:128]
            iv = i_emb.flatten()[:128]
            m_norms.append(mv / (torch.norm(mv) + 1e-8))
            i_norms.append(iv / (torch.norm(iv) + 1e-8))
            file_list.append(f)
            if len(file_list) < 5:
                print(f"{f} Input[0:5]: {iv[:5].tolist()}")
    except:
        pass

print(f"Total files: {len(file_list)}")
if len(file_list) > 1:
    min_m = 1; max_m = 0
    min_i = 1; max_i = 0
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            m_sim = torch.dot(m_norms[i], m_norms[j]).item()
            i_sim = torch.dot(i_norms[i], i_norms[j]).item()
            min_m = min(min_m, m_sim); max_m = max(max_m, m_sim)
            min_i = min(min_i, i_sim); max_i = max(max_i, i_sim)
    print(f"Memory Similarity: Min={min_m:.6f}, Max={max_m:.6f}")
    print(f"Input Similarity:  Min={min_i:.6f}, Max={max_i:.6f}")
