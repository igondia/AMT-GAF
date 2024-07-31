# --------------------------------------------------------
# AMT-GAF
# Licensed under BSD-2 License
# Written by Iván González
# --------------------------------------------------------

from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import pdb
from seqmodels.myTransformer import myTransformerEncoder,myTransformerEncoderLayer

class PositionalEncoding(nn.Module):
    # For this first version we just use the PE defined in the Attention is all you need paper (sin and cos)
    def __init__(self, d_model, dropout=0.1, max_len=5000, fusionType='sum'):
        """
            d_model: expected dimension of the encoder (d_encoder: 82, 256, 512 or 1024)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encoding
        pe = torch.zeros(1,max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term) # d_model (num features) needs to be even!!
        
        #Constant
        self.register_buffer('pe', pe)
        self.fusionType=fusionType
        if self.fusionType=='sum':
            self.scale_factor = nn.Parameter(torch.FloatTensor([1e-5]))
        elif self.fusionType=='concat':
            self.scale_factor = nn.Parameter(torch.FloatTensor([1.0]))
            
    def forward(self, x):
        if self.fusionType=='sum':
            x = x + self.scale_factor*self.pe[:,:x.shape[1], :]
        elif self.fusionType=='concat':
            x = torch.concatenate((x,self.scale_factor*self.pe[:,:x.shape[1], :]),dim=-1)
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_size,att_size,output_size,dropout=0.0, depth=1,nhead=1,temporal_horizon=0,pe_fusion_type='sum',pe_length=128):
        """
            num_class: number of classes
            input_size: number of input features
            att_size: size of the attention unit
            output_size: outputs size
            dropout: dropout probability
            depth: number of stacked encoder layers
            sequence_completion: if the network should be arranged for sequence completion pre-training
            return_context: whether to return the Rolling LSTM hidden and cell state (useful for MATT) during forward
        """
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.att_size = att_size 
        self.output_size = output_size
        self.nlayers=depth;
        self.temporal_horizon =temporal_horizon
        
        
        self.dp = nn.Dropout(p=dropout)
        #Copy
        self.linear_embedding=nn.Linear(input_size, att_size)
        if pe_fusion_type=='sum':
            self.pos_encoder = PositionalEncoding(att_size, dropout=dropout,max_len=5000,fusionType=pe_fusion_type) 
            encoder_layer=myTransformerEncoderLayer(att_size, nhead, dim_feedforward=output_size, dropout=dropout, batch_first=True)
            encoder_norm = nn.LayerNorm(att_size, eps=1e-05, elementwise_affine=True)
            
        elif pe_fusion_type=='concat':
            self.pos_encoder = PositionalEncoding(pe_length, dropout=dropout,max_len=5000,fusionType=pe_fusion_type)
            encoder_layer=myTransformerEncoderLayer(att_size+pe_length, nhead, dim_feedforward=output_size, dropout=dropout, batch_first=True)
            encoder_norm = nn.LayerNorm(att_size+pe_length, eps=1e-05, elementwise_affine=True)
            
        #encoder_norm is applied at the end of the encoder
        self.transformer = myTransformerEncoder(encoder_layer, self.nlayers, encoder_norm)
        
    def forward(self, inputs):
     
        device=inputs.device
        
        # Obtain mask with dimension n_obs x n_obs
        input_mask = self.generate_square_subsequent_mask(inputs.shape[1],length=self.temporal_horizon).to(device) # torch.size([n_obs, n_obs])
        
        #Linear embedding of the inputs    
        x = self.linear_embedding(inputs)
        # x = inputs
        #Add positional encoding
        x = self.pos_encoder(x)
        
        #Transformer
        x = self.transformer(x,input_mask)
        
        return x

    def generate_square_subsequent_mask(self, size,length=100):
          # mask future time observations (-inf) --> what about masking observations very far in the past ? config parameter
          mask = torch.zeros(size,size,dtype=torch.float)
          for i in range(size):
              iidx=np.maximum(i-length,0)
              mask[i,iidx:i+1]=1.0;
          mask=torch.log(mask);    
          
          return mask # torch.Size([size, size])
      
class MultiTaskModel(nn.Module):
    def __init__(self, seq_encoder, zlength,num_classes, gazeMapSize, fmapChannels=0,dropout=0.8, noise_std=0.1, nMap=9, mapOp='sum', packed=False):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            dropout: dropout probability
        """
        super(MultiTaskModel, self).__init__()
        self.seq_encoder = seq_encoder
        self.dp = nn.Dropout(dropout)
        
        self.zlength = zlength
        self.noise_std = noise_std
        self.nMap=nMap
        self.mapOp=mapOp;
        self.gazeMapSize=gazeMapSize
        self.packed=packed
        if fmapChannels>0:
            self.fmap_lin = nn.Conv2d(fmapChannels, 1,1,bias=True)
            
        #Gaze prediction layers
        self.weights_lin = nn.Linear(self.zlength, self.nMap, bias=True)
        self.mapNormalizer = torch.nn.InstanceNorm2d(1,affine=True)
        self.mapNormalizer1 = torch.nn.InstanceNorm2d(1,affine=True)
        self.mapNormalizer2 = torch.nn.InstanceNorm2d(1,affine=True)
        
        if mapOp=='1x_conv1x1':
            self.fix_conv1= nn.Conv2d(self.nMap, 1,1,bias=True)
        elif mapOp=='1x_conv3x3':
            self.fix_conv1= nn.Conv2d(self.nMap, 1,3,padding=1,bias=True)
        elif mapOp=='2x_conv3x3':
            self.fix_conv1= nn.Conv2d(self.nMap, 32,3,padding=1,bias=True)
            self.fix_conv2= nn.Conv2d(32,1,3,padding=1,bias=True)
        
        self.fc_gaze = nn.Linear(self.gazeMapSize[0]*self.gazeMapSize[1], 2, bias=True)
        
        #action_scores
        self.action_scores_lin = nn.Linear(self.zlength, num_classes, bias=True)
        nn.init.normal_(self.action_scores_lin.weight,std=0.1)
        nn.init.constant_(self.action_scores_lin.bias,0.0)
        
        
    def forward(self, inputs,smaps,fmap=None,predict_gaze=True):
        
        """inputs: tuple containing the inputs to the single branches"""
        
        # Sequence encoder
        mu = self.seq_encoder(inputs)
        
        # """SAMPLING Z"""
        if self.training:
            noise = torch.torch.randn_like(mu,device=mu.device,requires_grad=False)
            #Adaptive noise dependent on the values of the features
            noise_std = self.noise_std*mu.abs().mean()
            z = self.dp(mu + noise_std*noise)
        else:
            z = mu
            
        """"GENERATING GAZE PREDICTION"""
        map_weights = F.softmax(self.weights_lin(z),dim=2)
        map_weights_r = map_weights.view(map_weights.shape[0],map_weights.shape[1],map_weights.shape[2], 1,1)
        if fmap is not None:
            (numV,frs,tMaps,h,w)=fmap.shape
            gmap=self.fmap_lin(fmap.view(numV*frs,tMaps,h,w))
            gmap=gmap.view(numV,frs,1,h,w)
            smaps=torch.cat((smaps,gmap),2)
        (numV,frs,tMaps,h,w)=smaps.shape
         
        #Trick => We standardize the whole set of maps together to avoid incongruences between maps
        smaps=self.mapNormalizer(smaps.view(numV*frs,1,tMaps,h*w)).view(numV,frs,tMaps,h,w)
        
        
        # Modulate the maps
        fixmap = map_weights_r*smaps
        #For visualization
        i_fixmap = fixmap.clone()
        
        #Compute the final fixmap using the mix-model    
        (numV,frs,tMaps,h,w)=fixmap.shape
        #Generating the last fixmap
        if self.mapOp=='sum':
            fixmap = fixmap.sum(dim=2)
            fixmap=fixmap.view(numV,frs,-1);
        elif self.mapOp=='1x_conv3x3':
            fixmap=self.fix_conv1(fixmap.view(numV*frs,tMaps,h,w))
            fixmap=F.relu(self.mapNormalizer1(fixmap.view(numV*frs,1,fixmap.shape[1],h*w)).view(numV*frs,-1,h,w))
            fixmap=fixmap.view(numV,frs,-1);
        elif self.mapOp=='2x_conv3x3':
            #First convolutional block
            fixmap=self.fix_conv1(fixmap.view(numV*frs,tMaps,h,w))
            fixmap=F.relu(self.mapNormalizer1(fixmap.view(numV*frs,1,fixmap.shape[1],h*w)).view(numV*frs,-1,h,w))
            #Second convolutional block
            fixmap=self.fix_conv2(fixmap)
            fixmap=F.relu(self.mapNormalizer2(fixmap.view(numV*frs,1,fixmap.shape[1],h*w)).view(numV*frs,-1,h,w))
            #Final shape
            fixmap=fixmap.view(numV,frs,-1);
            
        #Regress the coordinates    
        fixation = self.fc_gaze(fixmap)
        fixation=torch.clip(fixation,-1,1)
        
        # Generate interpretable variable about gaze
        ifixation = (0.5*(fixation+0.999999)*torch.tensor((w,h),device=fixation.device)).to(torch.long)
        interpretable_w=i_fixmap[:,np.arange(frs),:,ifixation[...,1],ifixation[...,0]]
        interpretable_w=interpretable_w[:,:,0,:]
        #We put all values in positive => like denormalization
        interpretable_w=interpretable_w-interpretable_w.min()
        
        
        """PREDICTING ACTIONS"""
        action_scores =  self.action_scores_lin(self.dp(z))    

        return action_scores,fixation, interpretable_w 

            
       
class AMTGAF(nn.Module):
    def __init__(self,input_size, attention_size, zlength, num_classes,gazeMapSize,
                 depth=1, nhead=1,fmapChannels=0,dropout=0.8, noise_std=0.1, nMap=9,mapOp='sum',temporal_horizon=0,pe_fusion_type='sum',pe_length=128):
    
        super(AMTGAF, self).__init__() #Como Net es una clase hijo  de nn.Module, invocamos el constructor de nn.Module
        #Sequential Model
        seq_model=TransformerEncoder(input_size, attention_size, zlength,
                                         dropout=0,depth=depth,nhead=nhead,temporal_horizon=temporal_horizon,pe_fusion_type=pe_fusion_type,pe_length=pe_length)   
        #Multi-task model    
        self.model = MultiTaskModel(seq_model, zlength, num_classes, gazeMapSize, fmapChannels=fmapChannels, dropout=dropout, 
                                    noise_std=noise_std, nMap=nMap,mapOp=mapOp)
        
    def forward(self, inputs,smaps,fmap=None,predict_gaze=True):
        
        preds,fixations,interpretable_w = self.model(inputs,smaps,fmap,predict_gaze=predict_gaze)
        return {'act_preds' : preds,  'fixations' : fixations, 'interpretable_w' : interpretable_w}
    
    
   
