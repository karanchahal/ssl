import torch 
import numpy as np


def _simclr(im, aug_im):
    n, _ = im.size()
    im_2_aug = torch.mm(im, torch.transpose(aug_im, 0, 1)) # im = n x 128, 128 x n =>  n x n, where diagonal is positive, rest all are negatives
    im_2_other_im = torch.mm(im, torch.transpose(im, 0, 1)) # n x 128, 128 x n , -> n x n where diagonal is positive, rest all are negatives

    get_negs = torch.ones((n,n)).cuda()
    ind = torch.tensor(np.diag_indices(get_negs.shape[0]))
    get_negs[ind[0], ind[1]] = 0
    '''
    0 b c
    d 0 e
    f g 0
    '''
    one = get_negs * im_2_aug
    two = get_negs * im_2_other_im

    sigmas_neg = one.sum(dim=1) + two.sum(dim=1)

    get_pos = torch.zeros((n,n)).cuda()
    ind = np.diag_indices(get_pos.shape[0])
    get_pos[ind[0], ind[1]] = 1
    '''
    1 0 0
    0 1 0
    0 0 1
    '''

    one = get_pos * im_2_aug

    sigmas_pos = one.sum(dim=1) # get positive examples

    loss = sigmas_pos / (sigmas_neg + sigmas_pos)

    return loss


def analytical_ans():
    pos = [
        1*5 + 2*6,
        3*7 + 4*8
    ]

    negs = [
        1*(5+3+7) + 2*(6+4+8), # denominator of image 1 (negative samples + 1 positive sample)
        3*(1+5+7) + 4*(2+6+8), # denominator of image 2 (negative samples + 1 positive sample)
        5*(1+3+7) + 6*(2+4+8), # denominator of augmented image 1 (negative samples + 1 positive sample)
        7*(1+5+3) + 8*(2+6+4) # denominator of augmented image 2 (negative samples + 1 positive sample)
    ]


    ans = [
        pos[0] / negs[0],
        pos[1] / negs[1],
        pos[0]/ negs[2],
        pos[1]/ negs[3]
    ]

    return ans

def get_loss(a, b):
    loss1 = _simclr(a, b)
    loss2 = _simclr(b, a)
    loss = torch.sum(loss1 + loss2, dim=0)
    N = a.shape[0]
    return loss / (2*N)

def test_loss():

    a = torch.tensor([[1, 2], [3, 4]]).view(2,2)
    b = torch.tensor([[5, 6], [7, 8]]).view(2,2)
    
    loss1 = get_loss(a, b)
    loss2 = get_loss(b, a)

    loss = torch.cat((loss1, loss2), dim=0)
