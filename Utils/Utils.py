import torch as t
import torch.nn.functional as F
import numpy as np

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calcReward(bprLossDiff, keepRate):
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -t.log(nume / deno).mean()

def convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = t.Tensor(coo.row).long()
        col = t.Tensor(coo.col).long()
        index = t.stack([row, col])
        data = t.FloatTensor(coo.data)
        return t.sparse.FloatTensor(index, data, t.Size(coo.shape))

def knn_fast_threshold(user_embeddings, item_embeddings, b, T=0.1):
    num_users = user_embeddings.shape[0]
    num_items = item_embeddings.shape[0]
    
    index = 0

    # ユーザー-アイテムのエッジ情報を動的に格納するリスト
    values = []
    rows = []
    cols = []
    
    while index < num_users:
        end = min(index + b, num_users)
        
        # ユーザーのバッチを取得してアイテムとの類似度を計算
        sub_tensor = user_embeddings[index:end]
        similarities = t.mm(sub_tensor, item_embeddings.t())
        
        # 類似度を0から1の範囲に正規化
        probabilities = F.softmax(similarities, dim=1)
        
        # 閾値T以上の類似度のインデックスを取得
        above_threshold = probabilities >= T
        vals = probabilities[above_threshold]
        row_inds, col_inds = t.nonzero(above_threshold, as_tuple=True)
        
        # グローバルインデックスに変換
        rows.append(row_inds + index)  # ユーザーインデックス
        cols.append(col_inds + num_users)  # アイテムインデックス
        values.append(vals)  # 類似度スコア
        
        index += b
    
    # リストを1つのテンソルに結合
    rows = t.cat(rows) if rows else t.tensor([], device='cuda')
    cols = t.cat(cols) if cols else t.tensor([], device='cuda')
    values = t.cat(values) if values else t.tensor([], device='cuda')
    
    return rows, cols, values

def knn_fast(user_embeddings, item_embeddings, k, b):
    
    num_users = user_embeddings.shape[0]
    num_items = item_embeddings.shape[0]
    
    # ユーザー数 + アイテム数のサイズの正方行列のためのインデックス
    total_size = num_users + num_items
    index = 0
    
    # ユーザー-アイテムのエッジのみ格納
    values = t.zeros(num_users * k, device='cuda')
    rows = t.zeros(num_users * k, device='cuda')
    cols = t.zeros(num_users * k, device='cuda')
    norm_user = t.zeros(num_users, device='cuda')
    norm_item = t.zeros(num_items, device='cuda')
    
    while index < num_users:
        end = min(index + b, num_users)
        
        # ユーザーのバッチを取得してアイテムとの類似度を計算
        sub_tensor = user_embeddings[index:end]
        similarities = t.mm(sub_tensor, item_embeddings.t())
        
        # 各ユーザーに対してk個の最も近いアイテムを取得
        vals, inds = similarities.topk(k=k, dim=-1)
        
        # エッジの情報を保存
        values[index * k:end * k] = vals.view(-1)
        cols[index * k:end * k] = inds.view(-1) + num_users  # アイテムのインデックスを全体に対応させる
        rows[index * k:end * k] = t.arange(index, end, device='cuda').view(-1, 1).repeat(1, k).view(-1)
        
        
        index += b

    
    return rows, cols, values

def knn_dynamic(user_embeddings, item_embeddings, dynamic_k, b):
    """
    動的なk近傍エッジを計算する関数
    - user_embeddings: ユーザの埋め込みベクトル
    - item_embeddings: アイテムの埋め込みベクトル
    - dynamic_k: 各ユーザに対するエッジ数のリストまたはテンソル
    - b: バッチサイズ
    """
    num_users = user_embeddings.shape[0]
    num_items = item_embeddings.shape[0]
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

    # ユーザー数 + アイテム数のサイズの正方行列のためのインデックス
    total_size = num_users + num_items
    index = 0

    # リストに保存して最後に結合する
    values_list = []
    rows_list = []
    cols_list = []

    while index < num_users:
        end = min(index + b, num_users)

        # ユーザーのバッチを取得してアイテムとの類似度を計算
        sub_tensor = user_embeddings[index:end]
        similarities = t.mm(sub_tensor, item_embeddings.t())

        # 各ユーザに対して動的なk個の最も近いアイテムを取得
        for i, user_index in enumerate(range(index, end)):
            k = dynamic_k[user_index]  # このユーザに対する動的なkを取得
            vals, inds = similarities[i].topk(k=k, dim=-1)

            # エッジの情報を保存
            values_list.append(vals)
            cols_list.append(inds + num_users)  # アイテムのインデックスを全体に対応させる
            rows_list.append(t.full((k,), user_index, device='cuda'))

        index += b

    # リストを結合して最終的な結果を作成
    values = t.cat(values_list)
    rows = t.cat(rows_list)
    cols = t.cat(cols_list)

    return rows, cols, values

def add_edge(graph, item_embeddings, n_users, m_items, k, batch_size):
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
    num_items = item_embeddings.size(0)
    
    max_indices = []
    max_values = []
    
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_embeddings = item_embeddings[start:end]
        
        sim_matrix = t.mm(batch_embeddings, item_embeddings.t())
        
        # 対角成分を無視するために、対角成分を小さい値に設定
        sim_matrix.fill_diagonal_(-float('inf'))
        
        for i in range(k):
            value = t.max(sim_matrix)
            max_index = t.argmax(sim_matrix)  # フラットな1Dインデックスを取得

            # フラットなインデックスを2Dインデックスに変換
            indices = np.unravel_index(max_index.item(), sim_matrix.size())
            max_indices.append((indices[0] + start, indices[1]))  # バッチのオフセットを考慮
            max_values.append(value.item())
            sim_matrix[min(sim_matrix.shape[0] - 1,indices[0]), min(sim_matrix.shape[1] - 1, indices[1])] = -float('inf')
            sim_matrix[min(sim_matrix.shape[0] - 1, indices[1]), min(sim_matrix.shape[1] - 1, indices[0])] = -float('inf')
    
    # スパーステンソルのインデックスを取得
    indices = graph._indices()
    # ユーザとアイテムのインデックスを分離
    user_indices = indices[0]
    item_indices = indices[1] - n_users
    user_item_edges = {}
    for user in range(n_users):
        user_mask = user_indices == user
        user_item_edges[user] = item_indices[user_mask].tolist()
    
    # エッジの追加
    new_edges = []
    new_weights = []
    for i, (item1, item2) in enumerate(max_indices):
        for user, items in user_item_edges.items():
            if item1.item() in items and item2.item() not in items:
                new_edges.append((user, item2))
                new_weights.append(max_values[i])
            elif item2.item() in items and item1.item() not in items:
                new_edges.append((user, item1))
                new_weights.append(max_values[i])
    
    # 新しいエッジの行と列のインデックスと重みを返す
    if new_edges:
        new_user_indices, new_item_indices = zip(*new_edges)
        new_user_indices = t.tensor(new_user_indices, device=graph.device)
        new_item_indices = t.tensor(new_item_indices, device=graph.device) + n_users
        new_weights = t.tensor(new_weights, device=graph.device)
        return new_user_indices, new_item_indices, new_weights
    else:
        return t.tensor([], device=graph.device), t.tensor([], device=graph.device), t.tensor([], device=graph.device)
    
def add_edge2(graph, item_embeddings, n_users, m_items, k, batch_size):
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
    num_items = item_embeddings.size(0)
    
    # バッチごとの上位ペアを格納する
    partial_results = []
    
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_embeddings = item_embeddings[start:end]
        
        sim_matrix = t.mm(batch_embeddings, item_embeddings.t())
        # 類似度が1の要素を除外
        sim_matrix[sim_matrix == 1.0] = -float('inf')
        # 対角成分を無視
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # ここではバッチごとに、相当数の上位類似ペアを取得
        # 例として k の2倍を取得しておき、後で絞り込み
        local_k = k * 2  
        
        flat_vals = sim_matrix.view(-1)
        # バッチサイズが小さい場合、要素数が k 未満になる可能性があるので min(local_k, flat_vals.size(0)) を指定
        topk_vals, topk_indices = t.topk(
            flat_vals, 
            min(local_k, flat_vals.size(0)), 
            largest=True
        )

        # フラットなインデックスを2Dインデックスへ変換した上で partial_results に追加
        for val, idx in zip(topk_vals, topk_indices):
            i = idx // sim_matrix.shape[1]
            j = idx % sim_matrix.shape[1]
            partial_results.append((val.item(), i.item() + start, j.item()))
    
    # 全バッチから集めた結果を一括でソート
    partial_results.sort(key=lambda x: x[0], reverse=True)
    # グローバルに上位 k 個を取得
    final_results = partial_results[:k]
    
    # row, col, weight を作る
    row, col, weight = [], [], []
    for val, item1, item2 in final_results:
        row.append(item1 + n_users)
        col.append(item2 + n_users)
        weight.append(val)

    return (
        t.tensor(row).cuda(),
        t.tensor(col).cuda(),
        t.tensor(weight).cuda()
    )
    

def add_edge3(graph, item_embeddings, n_users, m_items, T, batch_size):
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
    num_items = item_embeddings.size(0)
    
    row = []
    col = []
    weight = []
    
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_embeddings = item_embeddings[start:end]
        
        sim_matrix = t.mm(batch_embeddings, item_embeddings.t())
        # 類似度が1の要素を除外
        sim_matrix[sim_matrix  >= 0.99999] = -float('inf')
        
        # 対角成分を無視するために、対角成分を小さい値に設定
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # 閾値T以上の類似アイテムペアを抽出
        indices = (sim_matrix >= T).nonzero(as_tuple=False)
        values = sim_matrix[indices[:, 0], indices[:, 1]]
        row.extend((indices[:, 0] + start + n_users).cpu().numpy())
        col.extend((indices[:, 1] + n_users).cpu().numpy())
        weight.extend(values.cpu().numpy())
    print(weight)
    print(T)
    
    return t.tensor(row).cuda(), t.tensor(col).cuda(), t.tensor(weight).cuda()
    
def normalize_adj(adj):
    # 行ごとの合計を計算
    rowsum = t.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = t.pow(rowsum, -0.5)
    d_inv_sqrt[t.isinf(d_inv_sqrt)] = 0.

    # スパースな対角行列 D^-0.5 を作成
    indices = t.arange(adj.shape[0], device=adj.device)
    d_mat_inv_sqrt = t.sparse_coo_tensor(
        t.stack([indices, indices]),
        d_inv_sqrt,
        adj.shape,
        device=adj.device
    )

    # 正規化隣接行列の計算: D^-0.5 * A * D^-0.5
    normalized_adj = t.sparse.mm(d_mat_inv_sqrt, adj)
    normalized_adj = t.sparse.mm(normalized_adj, d_mat_inv_sqrt)

    return normalized_adj

def get_feat_mask(num, mask_rate, latent_dim_rec):
     feat_node = latent_dim_rec
     mask = t.zeros((num), feat_node)
     samples = np.random.choice(feat_node, size=int(mask_rate * mask_rate), replace=False)
     mask[:, samples] = 1
     return mask.cuda(), samples

def calc_loss(x, x_aug, temperature=0.2, sym=True):
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = t.einsum('ik,jk->ij', x, x_aug) / t.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = t.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - t.log(loss_0).mean()
        loss_1 = - t.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - t.log(loss_1).mean()
        return loss_1

def calc_loss_gcl(anchor_user_x, anchor_item_x, user_x, item_x, temperature=0.2, batch_size=1024):
    """
    対照損失を計算する関数。
    
    Parameters:
    - anchor_user_x: アンカーユーザの埋め込み形状 (num_users, embedding_dim)
    - anchor_item_x: アンカーアイテムの埋め込み形状 (num_items, embedding_dim)
    - user_x: 全体のユーザ埋め込み
    - item_x: 全体のアイテム埋め込み
    - temperature: 温度パラメータ
    - batch_size: バッチサイズ

    Returns:
    - loss: 処理された対照損失（ユーザとアイテムの合計）
    """
    num_users = anchor_user_x.size(0)
    num_items = anchor_item_x.size(0)
    
    user_loss = 0.0
    item_loss = 0.0
    
    # ユーザの対照損失をバッチ処理
    for i in range(0, num_users, batch_size):
        end = min(i + batch_size, num_users)
        batch_anchor_user_x = anchor_user_x[i:end]
        batch_user_x = user_x[i:end]
        user_loss += calc_loss(batch_anchor_user_x, batch_user_x, temperature) * (end - i) / num_users
    
    # アイテムの対照損失をバッチ処理
    for i in range(0, num_items, batch_size):
        end = min(i + batch_size, num_items)
        batch_anchor_item_x = anchor_item_x[i:end]
        batch_item_x = item_x[i:end]
        item_loss += calc_loss(batch_anchor_item_x, batch_item_x, temperature) * (end - i) / num_items
    
    return user_loss + item_loss

def merge_sparse_adjs_union(adj_list):
    """
    複数の t.sparse.FloatTensor 形式の隣接行列を結合し、
    1つの隣接行列として返すサンプル関数です。

    同じエッジが複数の行列に存在する場合は加算されます。
    """
    if not adj_list:
        return None

    # 同じサイズの隣接行列が与えられることを想定
    shape = adj_list[0].shape
    indices_list = []
    values_list = []

    for adj in adj_list:
        # _indices(), _values() でスパーステンソルのインデックスと値を取得
        indices_list.append(adj._indices().cuda())
        values_list.append(adj._values().cuda())

    # cat で結合してから coalesce することで同一エッジをまとめる
    merged_indices = t.cat(indices_list, dim=1).cuda()
    merged_values = t.ones(len(t.cat(values_list, dim=0))).cuda()

    merged_adj = t.sparse_coo_tensor(
        merged_indices, merged_values, size=shape
    ).coalesce()
    return merged_adj

def merge_sparse_adjs_union2(adj_list):
    """
    複数の t.sparse.FloatTensor 形式の隣接行列を結合し、
    同じエッジ重複時は1つのエッジとして扱う (加算はしない) 関数サンプルです。
    """
    if not adj_list:
        return None

    shape = adj_list[0].shape
    edge_set = set()
    for adj in adj_list:
        idx = adj._indices()
        for i in range(idx.size(1)):
            r, c = idx[0, i].item(), idx[1, i].item()
            edge_set.add((r, c))

    merged_idx = t.tensor(list(edge_set)).t().long()
    merged_val = t.ones(merged_idx.size(1), dtype=t.float32)

    merged_adj = t.sparse_coo_tensor(
        merged_idx, merged_val, size=shape
    ).coalesce()

    return merged_adj