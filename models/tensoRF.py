from .tensorBase import *

class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, init_rank, rank_inc, **kargs):
        self.init_rank=init_rank[2:]
        self.rank_inc=rank_inc[2:]
        self.current_rank=self.init_rank[:]
        self.epsilon=1e-6
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
    def rank_mask(self, mask_rank, max_rank, mask_dim1, mask_dim2, device):
        ans_one=torch.ones((1,mask_rank,mask_dim1,mask_dim2),device=device)
        ans_zero=self.epsilon*torch.ones((1,max_rank-mask_rank,mask_dim1,mask_dim2),device=device)
        ans=torch.cat((ans_one,ans_zero),dim=1)
        return ans.to(device)
    def update_current_rank(self, device):
        self.current_rank[0]+=self.rank_inc[0]
        self.current_rank[1]+=self.rank_inc[1]
        d_line_dim=[self.density_line_mask[i].size()for i in range(len(self.vecMode))]
        a_line_dim=[self.app_line_mask[i].size()for i in range(len(self.vecMode))]
        d_plane_dim=[self.density_plane_mask[i].size()for i in range(len(self.matMode))]
        a_plane_dim=[self.app_plane_mask[i].size()for i in range(len(self.matMode))]
        d_line_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[0], self.density_n_comp[i], d_line_dim[i][2], 1, device))
            for i in range(len(self.vecMode))
        ]
        a_line_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[1], self.app_n_comp[i], a_line_dim[i][2], 1, device))
            for i in range(len(self.vecMode))
        ]
        d_plane_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[0], self.density_n_comp[i], d_plane_dim[i][2], d_plane_dim[i][3], device))
            for i in range(len(self.matMode))
        ]
        a_plane_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[1], self.app_n_comp[i], a_plane_dim[i][2], a_plane_dim[i][3], device))
            for i in range(len(self.matMode))
        ]
        self.density_line_mask=torch.nn.ParameterList(d_line_mask).to(device)
        self.app_line_mask=torch.nn.ParameterList(a_line_mask).to(device)
        self.density_plane_mask=torch.nn.ParameterList(d_plane_mask).to(device)
        self.app_plane_mask=torch.nn.ParameterList(a_plane_mask).to(device)
    def downdate_current_rank(self, device, down_rate = 4):
        self.current_rank[0]-=self.rank_inc[0]*down_rate
        self.current_rank[1]-=self.rank_inc[1]*down_rate
        d_line_dim=[self.density_line_mask[i].size()for i in range(len(self.vecMode))]
        a_line_dim=[self.app_line_mask[i].size()for i in range(len(self.vecMode))]
        d_plane_dim=[self.density_plane_mask[i].size()for i in range(len(self.matMode))]
        a_plane_dim=[self.app_plane_mask[i].size()for i in range(len(self.matMode))]
        d_line_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[0], self.density_n_comp[i], d_line_dim[i][2], 1, device))
            for i in range(len(self.vecMode))
        ]
        a_line_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[1], self.app_n_comp[i], a_line_dim[i][2], 1, device))
            for i in range(len(self.vecMode))
        ]
        d_plane_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[0], self.density_n_comp[i], d_plane_dim[i][2], d_plane_dim[i][3], device))
            for i in range(len(self.matMode))
        ]
        a_plane_mask=[
            torch.nn.Parameter(self.rank_mask(
                self.current_rank[1], self.app_n_comp[i], a_plane_dim[i][2], a_plane_dim[i][3], device))
            for i in range(len(self.matMode))
        ]
        self.density_line_mask=torch.nn.ParameterList(d_line_mask).to(device)
        self.app_line_mask=torch.nn.ParameterList(a_line_mask).to(device)
        self.density_plane_mask=torch.nn.ParameterList(d_plane_mask).to(device)
        self.app_plane_mask=torch.nn.ParameterList(a_plane_mask).to(device)
        self.density_line = torch.nn.ParameterList([torch.nn.Parameter(self.density_line[i]*self.density_line_mask[i]) for i in range(len(self.vecMode))])
        self.app_line = torch.nn.ParameterList([torch.nn.Parameter(self.app_line[i]*self.app_line_mask[i]) for i in range(len(self.vecMode))])
        self.density_plane = torch.nn.ParameterList([torch.nn.Parameter(self.density_plane[i]*self.density_plane_mask[i]) for i in range(len(self.matMode))])
        self.app_plane = torch.nn.ParameterList([torch.nn.Parameter(self.app_plane[i]*self.app_plane_mask[i]) for i in range(len(self.matMode))])
    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line, self.density_plane_mask, self.density_line_mask = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device, self.current_rank[0])
        self.app_plane, self.app_line, self.app_plane_mask, self.app_line_mask = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device, self.current_rank[1])
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
    def init_one_svd(self, n_component, gridSize, scale, device, mask_rank):
        plane_coef, line_coef = [], []
        plane_coef_mask, line_coef_mask=[], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))
            plane_coef_mask.append(torch.nn.Parameter(
                self.rank_mask(mask_rank, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0], device)))
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
            line_coef_mask.append(torch.nn.Parameter(
                self.rank_mask(mask_rank, n_component[i], gridSize[vec_id], 1, device)))
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device), \
                torch.nn.ParameterList(plane_coef_mask).to(device), torch.nn.ParameterList(line_coef_mask).to(device)
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars
    def vectorDiffs(self, vector_comps, dens_mark):
        total = 0
        for idx in range(len(vector_comps)):
            masked=vector_comps[idx]*self.density_line_mask[idx] if dens_mark \
                    else vector_comps[idx]*self.app_line_mask[idx]
            n_comp, n_size = masked.shape[1:-1]
            dotp = torch.matmul(masked.view(n_comp,n_size), masked.view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total
    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line,True) + self.vectorDiffs(self.app_line,False)
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx]*self.density_plane_mask[idx])) + \
                torch.mean(torch.abs(self.density_line[idx]*self.density_line_mask[idx]))
        return total
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]*self.density_plane_mask[idx]) * 1e-2
        return total
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]*self.app_plane_mask[idx]) * 1e-2
        return total
    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane]*self.density_plane_mask[idx_plane], 
                                                coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane]*self.density_line_mask[idx_plane],
                                                coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
        return sigma_feature
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane]*self.app_plane_mask[idx_plane], 
                                                  coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane]*self.app_line_mask[idx_plane], 
                                                 coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        return self.basis_mat((plane_coef_point * line_coef_point).T)
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef
    @torch.no_grad()
    def upsample_volume_grid(self, res_target, device):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        for i in range(len(self.vecMode)):
            d_mask_size=self.density_line_mask[i].size()
            a_mask_size=self.app_line_mask[i].size()
            d_dim=self.density_line[i].size()
            a_dim=self.app_line[i].size()
            self.density_line_mask[i]=self.rank_mask(self.current_rank[0],d_mask_size[1],d_dim[2],d_dim[3],device)
            self.app_line_mask[i]=self.rank_mask(self.current_rank[1],a_mask_size[1],a_dim[2],a_dim[3],device)
        for i in range(len(self.matMode)):
            d_mask_size=self.density_plane_mask[i].size()
            a_mask_size=self.app_plane_mask[i].size()
            d_dim=self.density_plane[i].size()
            a_dim=self.app_plane[i].size()
            self.density_plane_mask[i]=self.rank_mask(self.current_rank[0],d_mask_size[1],d_dim[2],d_dim[3],device)
            self.app_plane_mask[i]=self.rank_mask(self.current_rank[1],a_mask_size[1],a_dim[2],d_dim[3],device)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')
    @torch.no_grad() 
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)
        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.density_line_mask[i] = torch.nn.Parameter(
                self.density_line_mask[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line_mask[i] = torch.nn.Parameter(
                self.app_line_mask[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.density_plane_mask[i] = torch.nn.Parameter(
                self.density_plane_mask[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane_mask[i] = torch.nn.Parameter(
                self.app_plane_mask[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb
        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
