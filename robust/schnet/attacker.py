import tqdm
import torch as ch

import copy


def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch


class Attacker:
    """Performs an adversarial attack on an `initial` atom using the `ensemble` models."""
    def __init__(
        self,
        initial,
        ensemble,
        adv_loss,
        delta_init=0.01,
        epsilon=3,
        optim_lr=1e-2,
        device=0,
        nbr_list_update=2,
    ):
        self.initial = initial
        self.ensemble = ensemble
        self.loss_fn = adv_loss
        self.nbr_list_update = nbr_list_update
        
        self.delta_init = delta_init
        self.epsilon = epsilon
        self.optim_lr = optim_lr
        self.device = device
    
    @property
    def num_atoms(self):
        return len(self.initial)
    
    def initialize_translation(self, lattice=False):
        if lattice:
            delta = self.delta_init * ch.randn((3, 3), device=self.device)
        else:
            delta = self.delta_init * ch.randn((self.num_atoms, 3), device=self.device)
        delta.requires_grad = True
        opt = ch.optim.Adam([delta], lr=self.optim_lr)
        
        return delta, opt

    def attack(self, lattice=False, epochs=60):
        delta, opt = self.initialize_translation(lattice=lattice)
        
        results = []
        for epoch in tqdm.tqdm(range(epochs)):
            
            epoch_results = self.attack_epoch(opt, delta, epoch, lattice=lattice)
            results.append({
                'epoch': epoch,
                **epoch_results
            })
            
        return results
    
    def attack_epoch(self, opt, delta, epoch, lattice=False):
        opt.zero_grad()

        batch = self.prepare_attack(delta,
                                    update_nbr_list=(epoch % self.nbr_list_update == 0),
                                    lattice=lattice)

        results = [
            m(batch)
            for m in self.ensemble.models
        ]

        forces = ch.stack([
            r['energy_grad']
            for r in results
        ], dim=-1)

        if lattice:
            stresses = -ch.stack([
                r['stress']
                for r in results
            ], dim=-1)

        energy = ch.stack([
            r['energy']
            for r in results
        ], dim=-1)

        energy_per_atom = energy / self.num_atoms
        if lattice:
            loss = self.loss_fn.loss_fn(x=None, e=energy_per_atom, s=stresses).sum()
        else:
            loss = self.loss_fn.loss_fn(x=None, e=energy_per_atom, f=forces).sum()

        loss_item = loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-self.epsilon, self.epsilon)
        
        if lattice:    
            return {
                'delta': batch['delta'].clone().detach().cpu().numpy(),
                'energy': energy.detach().cpu().numpy(),
                'forces': forces.detach().cpu().numpy(),
                'stresses': stresses.detach().cpu().numpy(),
                'loss': loss_item,
            }
        else:
            return {
                'delta': batch['delta'].clone().detach().cpu().numpy(),
                'energy': energy.detach().cpu().numpy(),
                'forces': forces.detach().cpu().numpy(),
                'loss': loss_item,
            }
        

    def prepare_attack(self, delta, update_nbr_list=False, lattice=False):
        batch = self.get_batch(lattice=lattice)
        
        if lattice:
            batch['nxyz'].requires_grad = True
            batch['nxyz'] = ch.hstack([batch['nxyz'][:,:1],
                                        (batch['nxyz'][:,1:] + 
                                            ch.matmul(batch['nxyz'][:,1:], delta))]
                                     )

            # lattice_translation = self.get_translation(delta)
            batch['lattice'].requires_grad = True
            batch['lattice'] = batch['lattice'] + delta
        
        else:    
            nxyz_translation = self.get_translation(delta)
            
            # The following two lines allow the deltas to be backpropagated with the correct
            # neighbor list AND the attacked translation
            batch['nxyz'].requires_grad = True
            batch['nxyz'] = batch['nxyz'] + nxyz_translation
    
        if update_nbr_list:
            # the translation may change the neighbor list, so we update them every now and then
            nbr_list, offsets = self.update_nbr_list(delta, lattice=lattice)
            batch['nbr_list'] = nbr_list.to(self.device)
            batch['offsets'] = offsets.to(self.device) if isinstance(offsets, ch.Tensor) else offsets
        
        batch['delta'] = delta
        
        return batch
    
    def get_translation(self, delta):
        return ch.cat([
            ch.zeros((self.num_atoms, 1), device=delta.device),
            delta
        ], dim=1)
    
    def update_nbr_list(self, delta, lattice=False):
        # atoms = self.initial.copy()
        atoms = copy.deepcopy(self.initial)
        if lattice:
            atoms.set_cell(atoms.cell + delta.detach().cpu().numpy())
        else:
            atoms.translate(delta.detach().cpu().numpy())
        nbr_list, offsets = atoms.update_nbr_list()

        return nbr_list, offsets
    
    def get_batch(self, lattice=False):
        # atoms = self.initial.copy()
        atoms = copy.deepcopy(self.initial)
        batch = atoms.get_batch()

        batch['energy'] = 0
        batch['energy_grad'] = []
        if lattice:
            batch['stress'] = []
        batch['lattice'] = ch.tensor(atoms.get_cell().array)
        
        batch = batch_to(batch, self.device)
        return batch    
