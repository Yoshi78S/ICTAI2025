import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix
from graph_learner import UserItemGraph, graph_maker, graph_maker2, UserItemGraph2
import copy

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
		self.anchor_adj = copy.deepcopy(self.handler.torchBiAdj)
		self.anchor_adj = self.make_symmetric(self.anchor_adj)
		self.anchor_adj = self.anchor_adj.cuda()
		matrix = self.handler.trnMat.todense()
		self.interaction_num = [len(matrix[i].nonzero()[1]) + 2 for i in range(args.user)]

	def make_symmetric(self, adj):
        # 対称行列を作成
		return (adj + adj.transpose(0, 1)) / 2

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		log('Model Initialized')		
		topks = args.topk if isinstance(args.topk, list) else [args.topk]

		recallMax = {k: 0 for k in topks}
		ndcgMax = {k: 0 for k in topks}
		precisionMax = {k: 0 for k in topks}
		bestEpoch = {k: 0 for k in topks}		
		for ep in range(0, args.epoch):
				tstFlag = (ep % args.tstEpoch == 0)
				reses = self.trainEpoch(ep)
				log(self.makePrint('Train', ep, reses, tstFlag))		
				if tstFlag:
						reses = self.testEpoch()
						log(self.makePrint('Test', ep, reses, tstFlag))		
						for k in topks:
								r_key = f"Recall@{k}"
								n_key = f"NDCG@{k}"
								p_key = f"Precision@{k}"		
								if reses[r_key] > recallMax[k]:
										recallMax[k] = reses[r_key]
										ndcgMax[k] = reses[n_key]
										precisionMax[k] = reses[p_key]
										bestEpoch[k] = ep
				print()		
		for k in topks:
				print(f"Best epoch for top@{k} : {bestEpoch[k]}  | Recall@{k}: {recallMax[k]:.7f}  | NDCG@{k}: {ndcgMax[k]:.7f}  | Precision@{k}: {precisionMax[k]:.7f}")

	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		
		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			out_dims = eval(args.dims) + [args.item]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)
		
		self.item_features_txt = self.handler.text_feats.detach()
		self.user_features_txt = self.user_features(self.item_features_txt)
		self.item_features_txt = self.item_features_txt.cuda()
		self.user_features_txt = self.user_features_txt.cuda()
		feat_dim = self.item_features_txt.shape[1]
		#graph_learner_txt = UserItemGraph(args.n_ATT_layers, args.n_MLP_layers, args.hidden_size, feat_dim, feat_dim, args.k_txt, args.knn_metric, args.batch, args.mlp_act, args.T)
		self.graph_learner_txt = graph_maker2(args.user, args.item, self.item_features_txt, args.n_layers, args.T)
		self.graph_learner_txt = self.graph_learner_txt.cuda()
		#graph_learner_txt = UserItemGraph2(self.user_features_txt, self.item_features_txt, self.interaction_num, args.n_gcn_layers, args.knn_metric, args.batch, args.mlp_act)
		self.optimizer_txt = torch.optim.Adam(self.graph_learner_txt.parameters(), lr=args.lr, weight_decay=0)

		self.item_features_image = self.handler.image_feats.detach()
		self.user_features_image = self.user_features(self.item_features_image)
		self.item_features_image = self.item_features_image.cuda()
		self.user_features_image = self.user_features_image.cuda()
		feat_dim = self.item_features_image.shape[1]
		#graph_learner_image = UserItemGraph(args.n_ATT_layers, args.n_MLP_layers, args.hidden_size, feat_dim, feat_dim, args.k_img, args.knn_metric, args.batch, args.mlp_act, args.T)
		self.graph_learner_image = graph_maker2(args.user, args.item, self.item_features_image, args.n_layers, args.T)
		self.graph_learner_image = self.graph_learner_image.cuda()
		#graph_learner_image = UserItemGraph2(self.user_features_image, self.item_features_image, self.interaction_num, args.n_gcn_layers, args.knn_metric, args.batch, args.mlp_act)
		self.optimizer_image = torch.optim.Adam(self.graph_learner_image.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			self.item_features_audio = self.handler.audio_feats.detach()
			self.user_features_audio = self.user_features(self.item_features_audio)
			self.item_features_audio = self.item_features_audio.cuda()
			self.user_features_audio = self.user_features_audio.cuda()
			feat_dim = self.item_features_audio.shape[1]
			#graph_learner_audio = UserItemGraph(args.n_ATT_layers, args.n_MLP_layers, args.hidden_size, feat_dim, feat_dim, args.k_aud, args.knn_metric, args.batch, args.mlp_act, args.T)
			self.graph_learner_audio = graph_maker2(args.user, args.item, self.item_features_audio, args.n_layers, args.T)
			self.graph_learner_audio = self.graph_learner_audio.cuda()
			#graph_learner_audio = UserItemGraph2(self.user_features_audio, self.item_features_audio, self.interaction_num, args.n_gcn_layers, args.knn_metric, args.batch, args.mlp_act)
			self.optimizer_audio = torch.optim.Adam(self.graph_learner_audio.parameters(), lr=args.lr, weight_decay=0)


		

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	def user_features(self, item_features):
		user_features = []
		tmp_user = 0
		item_features_numpy = item_features.cpu().numpy()
		matrix = self.handler.trnMat.todense()
		for i in range(args.user):
			tmp_features = item_features_numpy[matrix[i].nonzero()[1]]
			tmp_features = np.mean(tmp_features, axis=0)
			user_features.append(tmp_features)
		user_features = np.array(user_features)
		user_features = torch.tensor(user_features, dtype=torch.float32)
		return user_features



	def trainEpoch(self, epoch):
		
		self.item_features_txt = self.handler.text_feats.detach()
		self.user_features_txt = self.user_features(self.item_features_txt)
		self.item_features_txt = self.item_features_txt.cuda()
		self.user_features_txt = self.user_features_txt.cuda()
		feat_dim = self.item_features_txt.shape[1]
		#graph_learner_txt = UserItemGraph(args.n_ATT_layers, args.n_MLP_layers, args.hidden_size, feat_dim, feat_dim, args.k_txt, args.knn_metric, args.batch, args.mlp_act, args.T)
		#graph_learner_txt = graph_maker2(args.user, args.item, self.item_features_txt, args.n_layers, args.T)
		#graph_learner_txt = UserItemGraph2(self.user_features_txt, self.item_features_txt, self.interaction_num, args.n_gcn_layers, args.knn_metric, args.batch, args.mlp_act)
		#optimizer_txt = torch.optim.Adam(graph_learner_txt.parameters(), lr=args.lr, weight_decay=0)
		
		self.item_features_image = self.handler.image_feats.detach()
		self.user_features_image = self.user_features(self.item_features_image)
		self.item_features_image = self.item_features_image.cuda()
		self.user_features_image = self.user_features_image.cuda()
		feat_dim = self.item_features_image.shape[1]
		#graph_learner_image = UserItemGraph(args.n_ATT_layers, args.n_MLP_layers, args.hidden_size, feat_dim, feat_dim, args.k_img, args.knn_metric, args.batch, args.mlp_act, args.T)
		#graph_learner_image = graph_maker2(args.user, args.item, self.item_features_image, args.n_layers, args.T)
		#graph_learner_image = UserItemGraph2(self.user_features_image, self.item_features_image, self.interaction_num, args.n_gcn_layers, args.knn_metric, args.batch, args.mlp_act)
		#optimizer_image = torch.optim.Adam(graph_learner_image.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			self.item_features_audio = self.handler.audio_feats.detach()
			self.user_features_audio = self.user_features(self.item_features_audio)
			self.item_features_audio = self.item_features_audio.cuda()
			self.user_features_audio = self.user_features_audio.cuda()
			feat_dim = self.item_features_audio.shape[1]
			#graph_learner_audio = UserItemGraph(args.n_ATT_layers, args.n_MLP_layers, args.hidden_size, feat_dim, feat_dim, args.k_aud, args.knn_metric, args.batch, args.mlp_act, args.T)
			#graph_learner_audio = graph_maker2(args.user, args.item, self.item_features_audio, args.n_layers, args.T)
			#graph_learner_audio = UserItemGraph2(self.user_features_audio, self.item_features_audio, self.interaction_num, args.n_gcn_layers, args.knn_metric, args.batch, args.mlp_act)
			#optimizer_audio = torch.optim.Adam(graph_learner_audio.parameters(), lr=args.lr, weight_decay=0)
		"""
		graph_learner_txt = graph_learner_txt.cuda()
		graph_learner_image = graph_learner_image.cuda()
		if args.data == 'tiktok':
			graph_learner_audio = graph_learner_audio.cuda()
		"""
		
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()

		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
		steps = trnLoader.dataset.__len__() // args.batch
		"""
		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()

			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
			if args.data == 'tiktok':
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats)

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()

			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()

			#log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item, args.sampling_steps, args.sampling_noise)
					top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)
			
			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			
			"""
		with torch.no_grad():
			k_img = int(float(args.k_img)/100.0 * float(self.handler.trnLoader.dataset.__len__()))
			k_txt = int(float(args.k_txt)/100.0 * float(self.handler.trnLoader.dataset.__len__()))
			if args.data == 'tiktok':
				k_aud = int(float(args.k_aud)/100.0 * float(self.handler.trnLoader.dataset.__len__()))
				print('k_txt', k_txt, 'k_img', k_img, 'k_aud', k_aud)
			else:
				print('k_txt', k_txt, 'k_img', k_img)
			

			user_original_embeddings = self.model.uEmbeds
			item_original_embeddings = self.model.iEmbeds
			#self.image_UI_matrix = graph_learner_image(self.user_features_image, self.item_features_image)
			#self.image_UI_matrix = graph_learner_image(self.anchor_adj, args.k_img, args.batch)
			self.image_UI_matrix = self.graph_learner_image(self.anchor_adj, k_img, item_original_embeddings, args.batch)
			#self.image_UI_matrix = graph_learner_image(self.handler.torchBiAdj)
			self.image_UI_matrix = normalize_adj(self.image_UI_matrix)
			self.image_UI_matrix = self.image_UI_matrix.cuda()
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)
			"""
			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			"""
			#self.text_UI_matrix = graph_learner_txt(self.user_features_txt, self.item_features_txt)
			#self.text_UI_matrix = graph_learner_txt(self.anchor_adj, args.k_txt, args.batch)
			self.text_UI_matrix = self.graph_learner_txt(self.anchor_adj, k_txt, item_original_embeddings, args.batch)
			#self.text_UI_matrix = graph_learner_txt(self.handler.torchBiAdj)
			self.text_UI_matrix = normalize_adj(self.text_UI_matrix)
			self.text_UI_matrix = self.text_UI_matrix.cuda()
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				"""
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				"""
				#self.audio_UI_matrix = graph_learner_audio(self.user_features_audio, self.item_features_audio)
				#self.audio_UI_matrix = graph_learner_audio(self.anchor_adj, args.k_aud, args.batch)
				self.audio_UI_matrix = self.graph_learner_audio(self.anchor_adj, k_aud, item_original_embeddings, args.batch)
				#self.audio_UI_matrix = graph_learner_audio(self.handler.torchBiAdj)
				self.audio_UI_matrix = normalize_adj(self.audio_UI_matrix)
				self.audio_UI_matrix = self.audio_UI_matrix.cuda()
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.anchor_adj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.anchor_adj, self.image_UI_matrix, self.text_UI_matrix)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			loss = bprLoss + regLoss
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			if args.data == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.anchor_adj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.anchor_adj, self.image_UI_matrix, self.text_UI_matrix)
			if args.data == 'tiktok':
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			else:
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			if args.data == 'tiktok':
				clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss_ = clLoss1 + clLoss2 + clLoss3
			else:
				clLoss_ = clLoss1 + clLoss2

			if args.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss
			CLLOSS = clLoss
			self.optimizer_txt.zero_grad()
			self.optimizer_image.zero_grad()
			if args.data == 'tiktok':
				self.optimizer_audio.zero_grad()
			CLLOSS.backward(retain_graph=True)
			self.optimizer_txt.step()
			self.optimizer_image.step()
			if args.data == 'tiktok':
				self.optimizer_audio.step()

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()
			"""
			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)
		"""
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		#ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		#ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		#if args.data == 'tiktok':
			#ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
		"""
		adj_list = []
		if args.data == 'tiktok':
			adj_list.append(self.image_UI_matrix)
			adj_list.append(self.text_UI_matrix)
			adj_list.append(self.audio_UI_matrix)
		else:
			adj_list.append(self.image_UI_matrix)
			adj_list.append(self.text_UI_matrix)
		if epoch % args.c == 0 and epoch != 0:
			self.anchor_adj = merge_sparse_adjs_union(adj_list)
			self.anchor_adj = normalize_adj(self.anchor_adj)
			self.anchor_adj = self.anchor_adj.cuda()
		"""
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		topks = args.topk if isinstance(args.topk, list) else [args.topk]
		epRecall = {k: 0 for k in topks}
		epNdcg = {k: 0 for k in topks}
		epPrecision = {k: 0 for k in topks}
		i = 0
		num = len(tstLoader.dataset)
		steps = num // args.tstBat		
		if args.data == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.anchor_adj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
		else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.anchor_adj, self.image_UI_matrix, self.text_UI_matrix)		
		for usr, trnMask in tstLoader:
				i += 1
				usr = usr.long().cuda()
				trnMask = trnMask.cuda()
				allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
				max_topk = max(topks)
				_, topLocs = torch.topk(allPreds, max_topk)
				recalls, ndcgs, precisions = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr.cpu().numpy(), topks)
				for k in topks:
						epRecall[k] += recalls[k]
						epNdcg[k] += ndcgs[k]
						epPrecision[k] += precisions[k]		
		ret = dict()
		for k in topks:
				ret[f'Recall@{k}'] = epRecall[k] / num
				ret[f'NDCG@{k}'] = epNdcg[k] / num
				ret[f'Precision@{k}'] = epPrecision[k] / num
		return ret


	def calcRes(self, topLocs, tstLocs, batIds, topks):	
		allRecall = {k: 0 for k in topks}
		allNdcg = {k: 0 for k in topks}
		allPrecision = {k: 0 for k in topks}		
		for i in range(len(batIds)):
				temTopLocs = list(topLocs[i])
				temTstLocs = tstLocs[batIds[i]]
				tstNum = len(temTstLocs)		
				for k in topks:
						topKLocs = temTopLocs[:k]
						maxDcg = np.sum([1 / np.log2(j + 2) for j in range(min(tstNum, k))])
						recall = dcg = precision = 0
						for val in temTstLocs:
								if val in topKLocs:
										recall += 1
										dcg += 1 / np.log2(topKLocs.index(val) + 2)
										precision += 1
						recall = recall / tstNum
						ndcg = dcg / maxDcg if maxDcg > 0 else 0
						precision = precision / k
						allRecall[k] += recall
						allNdcg[k] += ndcg
						allPrecision[k] += precision		
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()