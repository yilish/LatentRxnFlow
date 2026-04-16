import numpy as np
import torch
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw
from PIL import Image
from PIL import ImageDraw
from PIL import ImageDraw
MAX_BONDS = 6
MAX_DIFF = 4
def create_gradient(start_color, end_color, num_steps=256):
    """
    Create a gradient from start_color to end_color.

    Parameters:
    - start_color: Tuple of RGB values (e.g., (255, 0, 51) for #ff0033).
    - end_color: Tuple of RGB values (e.g., (248, 155, 41) for #f89b29).
    - num_steps: Number of steps in the gradient.

    Returns:
    - A list of RGB tuples representing the gradient.
    """
    gradient = [
        (
            int(start_color[0] + (end_color[0] - start_color[0]) * i / (num_steps - 1)),
            int(start_color[1] + (end_color[1] - start_color[1]) * i / (num_steps - 1)),
            int(start_color[2] + (end_color[2] - start_color[2]) * i / (num_steps - 1))
        )
        for i in range(num_steps)
    ]
    return gradient
#%%
def mol2array(mol, bond_labels=None):
    """
    将分子转换为 NumPy 数组，并在 bond 上添加额外信息（例如标签）。
    
    参数：
    - mol: RDKit 分子对象
    - bond_labels: 字典，键为 (atom1, atom2) 的元组，值为要在 bond 上绘制的字符串
    
    返回：
    - array: 形状为 (H, W, 3) 的 NumPy 数组
    """
    # 生成分子图片
    size= (600, 400)
    img = Draw.MolToImage(mol, kekulize=False, size=size)
    img = img.convert('RGBA')

    draw = ImageDraw.Draw(img, mode='RGBA')

    # if bond_labels:
        # 获取绘制坐标
    mol_draw = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    mol_draw.DrawMolecule(mol)
    mol_draw.FinishDrawing()
    atom_positions = []
    # generate gradient color from red to blue
    # create gradient from #ff0033 -> #f89b29 in RGB256
    colors = create_gradient((255, 0, 51), (248, 155, 41), 5)
    for atom in mol.GetAtoms():
        
        atom_positions.append(mol_draw.GetDrawCoords(atom.GetIdx()))
    #     try:
    #         # Parse top-k value and calculate alpha
    #         top_k_value = int(atom.GetProp('delta_bond').split(' ')[1].split(':')[0])
            
            
    #         # Use RGBA color with calculated alpha
    #         text_color = (128, 128, 128)
    #         circle_color = colors[top_k_value-1]
            
    #         draw.text((atom_positions[-1].x, atom_positions[-1].y), atom.GetProp('delta_bond'), fill=text_color)
    #         draw.ellipse(
    #             (atom_positions[-1].x - 15, atom_positions[-1].y - 15, 
    #              atom_positions[-1].x + 15, atom_positions[-1].y + 15), 
    #             outline=circle_color, width=2, fill=None
    #         )
    #     except:
    #         pass
        # atom.GetProp('delta_bond', f"{float((delta_bond[i, j]+delta_bond[j, i])/2):.3f}")

    # 在 bonds 上绘制文本
    

    # 创建一个新的透明图层
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # 在透明图层上绘制内容
    for atom in mol.GetAtoms():
        if atom.GetIdx() == 6 or atom.GetIdx() == 8:
            # top_k_value = int(atom.GetProp('delta_bond').split(' ')[1].split(':')[0])
            # alpha = max(0, min(128, int(128 * (1 - (top_k_value - 1) / 8))))  # Scale alpha based on k (1-5)
            circle_color = (255, 0, 0, 128)
            
            position = atom_positions[atom.GetIdx()]
            overlay_draw.ellipse(
                (position.x - 20, position.y - 20, position.x + 20, position.y + 20),
                outline=circle_color, width=5
            )
        if atom.HasProp('delta_bond'):
            top_k_value = int(atom.GetProp('delta_bond').split(' ')[1].split(':')[0])
            alpha = max(0, min(128, int(128 * (1 - (top_k_value - 1) / 8))))  # Scale alpha based on k (1-5)
            circle_color = (255, 0, 0, alpha)
            if top_k_value == 1:
                circle_color = (255, 0, 0, 255)
            position = atom_positions[atom.GetIdx()]
            overlay_draw.ellipse(
                (position.x - 10, position.y - 10, position.x + 10, position.y + 10),
                outline=circle_color, width=2, fill=circle_color
            )
        
    for bond in mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mid_x = (atom_positions[idx1].x + atom_positions[idx2].x) / 2 + 10
        mid_y = (atom_positions[idx1].y + atom_positions[idx2].y) / 2
        # try:
        overlay_draw.text((mid_x, mid_y), bond.GetProp('delta_bond'), fill=(255, 0, 0))
        # except:
            # pass
    # 合并透明图层和原始图像
    img = Image.alpha_composite(img, overlay)

    # 转换为 NumPy 数组
    array = np.array(img)#[:, :, 0:3]
    return array


def check(smile):
    smile = smile.split('.')
    smile.sort(key = len)
    try:
        mol = Chem.MolFromSmiles(smile[-1], sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except Exception:
        return False

def mol2file(m, name):
    AllChem.Compute2DCoords(m)
    img = Draw.MolToImage(m)
    Draw.MolToFile(m, os.path.join('./img', name))


def result2mol_transfer(args): # for threading
    element, mask, bond, aroma, charge, reactant, delta_bond = args
    # [L], [L], [L, 4], [l], [l]
    mask = mask.ne(1)
    cur_len = sum(mask.long())
    l = element.shape[0]

    mol = Chem.RWMol()
    
    element = element.cpu().numpy().tolist()
    charge = charge.cpu().numpy().tolist()
    bond = bond.cpu().numpy().tolist()    
    
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(l):
        if mask[i] == False:
            continue
        a = Chem.Atom(element[i])
        if not reactant is None and reactant[i]:
            a.SetAtomMapNum(i+1)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
    
    # add bonds between adjacent atoms
    # count = 0
    for this in range(l):
        if mask[this] == False:
            continue
        lst = bond[this]
        
        for j in range(len(bond[0])):
            other = bond[this][j]
            # only traverse half the matrix
            if other >= this or other in lst[0:j] or not this in bond[other]:
                continue
            if lst.count(other)==3 or bond[other].count(this) == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type) 
            elif lst.count(other) == 2 or bond[other].count(this) == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)   
            else:
                if aroma[this]==aroma[other] and aroma[this]>0: 
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
            
            
            if float(abs(delta_bond[this, other].item())) > 0.001:
                if element[this] != 35   and element[other] != 35:
                    continue
                prob = float((delta_bond[this, other].item()+delta_bond[other, this].item())/2)

                mol.GetBondBetweenAtoms(node_to_idx[this], node_to_idx[other]).SetProp('delta_bond', f"{prob:.3f}")
                mol.GetBondBetweenAtoms(node_to_idx[other], node_to_idx[this]).SetProp('delta_bond', f"{prob:.3f}")
                print(element[this], element[other], prob)
        # scan delta_bond[i, j]
                if abs(prob) > 0.1:
                    # select top 5 argsort
                    # transfer bond[this] to bit mask
                    delta = torch.abs(delta_bond[this])+torch.abs(delta_bond[:, this])
                    bond_mask = torch.ones(l).cuda()
                    bond_mask[bond[this]] = 0
                    bond_mask[this] = 0
                    delta = delta * bond_mask
                    
                    softmax_prob = torch.softmax(delta, dim=0)
                    # continue
                    top_5 = torch.flip(torch.argsort(softmax_prob)[-8:], dims=[0]).tolist()
                    print(top_5)
                    for top, j in enumerate(top_5):
                        if this == j:
                            continue
                        if mol.GetBondBetweenAtoms(node_to_idx[this], node_to_idx[j]) is not None:
                            continue
                        # if j in lst:
                        #     continue
                        # if element[j] != 6:
                        #     continue
                        
                        if (abs(delta_bond[this, j]) > 0.001 or abs(delta_bond[j, this]) > 0.001) and mask[this] and mask[j]:
                        
                            mol.GetAtomWithIdx(node_to_idx[j]).SetProp('delta_bond', f"Top {top+1}: {float(softmax_prob[j]):.3f}")
                            # mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.rdchem.BondType.SINGLE)
                            # mol.GetBondBetweenAtoms(node_to_idx[i], node_to_idx[j]).SetProp('delta_bond', f"{float((delta_bond[i, j]+delta_bond[j, i])/2):.3f}")
    # for i in range(l):
    #     for j in range(l):
    #         if i == j:
    #             continue
    #         if abs(delta_bond[i, j]) > 0.01 and mask[i] and mask[j]:
    #             if mol.GetBondBetweenAtoms(node_to_idx[i], node_to_idx[j]) is not None:
    #                 mol.GetBondBetweenAtoms(node_to_idx[i], node_to_idx[j]).SetProp('delta_bond', f"{float((delta_bond[i, j]+delta_bond[j, i])/2):.3f}")
    #             else:
    #                 mol.GetAtomWithIdx(node_to_idx[i]).SetProp('delta_bond', f"{float((delta_bond[i, j]+delta_bond[j, i])/2):.3f}")
    #                 # mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.rdchem.BondType.SINGLE)
                    # mol.GetBondBetweenAtoms(node_to_idx[i], node_to_idx[j]).SetProp('delta_bond', f"{float((delta_bond[i, j]+delta_bond[j, i])/2):.3f}")
    for i, item in enumerate(charge):
        if mask[i] == False:
            continue
        if not item == 0:
            atom = mol.GetAtomWithIdx(node_to_idx[i])
            atom.SetFormalCharge(item)
    # Convert RWMol to Mol object
    mol = mol.GetMol() 
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    smile = Chem.MolToSmiles(mol)
    return mol, smile, check(smile)

def result2mol(args): # for threading
    element, mask, bond, aroma, charge, reactant = args
    # [L], [L], [L, 4], [l], [l]
    mask = mask.ne(1)
    cur_len = sum(mask.long())
    l = element.shape[0]

    mol = Chem.RWMol()
    
    element = element.cpu().numpy().tolist()
    charge = charge.cpu().numpy().tolist()
    bond = bond.cpu().numpy().tolist()    
    
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(l):
        if mask[i] == False:
            continue
        a = Chem.Atom(element[i])
        if not reactant is None and reactant[i]:
            a.SetAtomMapNum(i+1)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for this in range(l):
        if mask[this] == False:
            continue
        lst = bond[this]
        for j in range(len(bond[0])):
            other = bond[this][j]
            # only traverse half the matrix
            if other >= this or other in lst[0:j] or not this in bond[other]:
                continue
            if lst.count(other)==3 or bond[other].count(this) == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type) 
            elif lst.count(other) == 2 or bond[other].count(this) == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)   
            else:
                if aroma[this]==aroma[other] and aroma[this]>0: 
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
                 
    for i, item in enumerate(charge):
        if mask[i] == False:
            continue
        if not item == 0:
            atom = mol.GetAtomWithIdx(node_to_idx[i])
            atom.SetFormalCharge(item)
    # Convert RWMol to Mol object
    mol = mol.GetMol() 
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    smile = Chem.MolToSmiles(mol)
    return mol, smile, check(smile)

def visualize(element, mask, bond, aroma, charge, reactant=None, delta_bond=None):
    mol, smile, _ = result2mol_transfer((element, mask, bond, aroma, charge, reactant, delta_bond))
    array = mol2array(mol)
    return array, smile


#%%
# def raw_bond_to_adj(raw_bond)
def vis_d(d, sample):
    # print(d['raw_data'].keys())
    d_rawdata = d['raw_data']
    element, mask, bond, aroma, charge, reactant = d_rawdata['element'][sample], d_rawdata['src_mask'][sample], d_rawdata['src_bond'][sample], d_rawdata['src_aroma'][sample], d_rawdata['src_charge'][sample], d_rawdata['src_charge'][sample]
    # print(d['bond'].shape)
    pred_charge = d['charge'][sample].argmax(dim=0) - 6
    pred_aroma = d['aroma'][sample].argmax(dim=0)#.cpu().numpy()
    # print(pred_charge, pred_aroma)
    # break
    delta_bond = d['bond'][sample]#.cpu().numpy()
    array, smile = visualize(element, mask, bond, aroma, charge, reactant, delta_bond)
    
    # arrays.append(array)
    return array
    
