

import re
import numpy as np
from ase.data import chemical_symbols


class Gaussian16:

   def __repr__(self):
      return 'Extract information from Gaussian16 outputs'

   def __init__(self, output, work_type):
         self.work_type = work_type
         self.nosym = False
         self.lines = []

         with open(output, 'r') as file:
            lines = file.readlines()
         if  'opt' in work_type:
            for line in reversed(lines):
                  if 'nosym' in line:
                     self.nosym = True
                  self.lines.append(line)
                  if work_type == 'opt+freq':
                     strs='Input orientation'
                  elif work_type == 'opt':
                     strs='SCF Done'
                  if strs in line:
                     break
            self.lines.reverse()
         elif work_type == 'sp':
            self.lines = lines

   @property
   def get_atom_labels(self):
      if self.work_type == 'opt' and self.nosym==False:
         strs='Standard orientation'
      else:
         strs='Input orientation'
      atoms=[]
      match = False
      count = 0
      for line in self.lines:
         if strs in line:
            match = True
            count += 1
            continue
         elif match and (count >= 5):
            if len(line.split()) != 6:
               break
            atoms.append(int(line.split()[1]))
         elif match:
            count += 1
      # labels = np.asarray([atomic_number_to_element.get(number) for number in atoms])
      labels = np.asarray([chemical_symbols[number] for number in atoms])
      labels = np.expand_dims(labels, axis=0)
      return labels

   @property
   def get_coords(self):
      xyz_list = list()
      match = False
      count = 0

      if self.work_type == 'opt' and self.nosym==False:
         strs='Standard orientation'
      else:
         strs='Input orientation'

      for line in self.lines:
         if strs in line:
            match = True
            count += 1
            continue
         elif match and (count >= 5):
            if len(line.split()) != 6:
               break
            xyz = np.array(line.split()[-3:], dtype=str)
            xyz = xyz.reshape(-1,3)
            xyz_list.append(xyz)
         elif match:
            count += 1
      coords = np.vstack(xyz_list)
      coords=np.array(coords, dtype=np.float64)
      # coords = np.expand_dims(coords, axis=0)
      return coords

   @property
   def get_forces(self):
      start_reading = False
      count = 0
      F_list = list()
      for line in self.lines:
         if start_reading:
            count += 1
         if 'Forces (Hartrees/Bohr)' in line:
            start_reading = True
            count += 1
            continue
         elif start_reading and (count >= 4):
            if len(line.split()) != 5:
               break
            xyz = np.array(line.split()[-3:], dtype=np.float64)
            xyz = xyz.reshape(-1,3)
            F_list.append(xyz)
      forces = np.vstack(F_list)

      # ua_to_kcal = 1185.8212143783846
      # forces *= ua_to_kcal

      formatted_forces = np.char.mod('%0.9f', forces)
      forces=np.array(formatted_forces, dtype=np.float64)
      # forces = np.expand_dims(forces, axis=0)
      return forces

   @property
   def get_total_energy(self):
      for line in self.lines:
         if 'SCF Done' in line:
            total_energy = np.array(line.split()[4], dtype=np.float64, ndmin=2)
            # output energy in kcal/mol
            total_energy *= 627.5096080305927
      return total_energy

   @property
   def get_homo_lumo(self):
      for line in self.lines:
         if ' occ. ' in line:
            e_homo = float(line.split()[-1])
         if ' virt. ' in line:
            e_lumo = float(line.split()[4])
            break
      e_homo_lumo = np.array([e_homo, e_lumo])
      e_homo_lumo = e_homo_lumo.reshape(-1,2)
      # convert from Hartree to eV
      e_homo_lumo *= 27.211399
      return e_homo_lumo

   @property
   def get_dipole_AuToDebye(self):
   #    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[D]\ *-?\ *[0-9]+)?')
      scinot = re.compile('[-+]?[\d]+\.?[\d]*[Dd](?:[-+]?[\d]+)?')
      debye_to_au = 0.393430
      for line in self.lines:
         if 'Dipole    ' in line:
            dipole_list=re.findall(scinot, line)
            if len(dipole_list) !=3:
               continue
            dipole = [x.replace('D','E') for x in dipole_list]
            # output the dipole in Debye
            dipole = np.array(dipole, dtype=np.float64) / debye_to_au
            dipole = dipole.reshape(1,3)
            return dipole

   @property
   def get_dipole(self):
   #Here we strightforwardly get dipole in Debye
      read_data = False
      for line in self.lines:
         if 'Dipole moment (field-independent basis, Debye)' in line:
            read_data = True
            continue
         if read_data == True:
            dipoles = list(map(float, line.split()[1:6:2]))[0:3]
            break
      dipole = np.array(dipoles, dtype=np.float64).reshape(1,3)
      return dipole

   @property
   def get_quadrupole(self):
      n_dim = 3
      quad_matrix = np.zeros((n_dim,n_dim), dtype=np.float64)
      read_data = False
      for line in self.lines:
         if re.match(r'^ Quadrupole moment', line):
            read_data = True
            continue
         if 'Traceless' in line:
            break
         if read_data:
            m_elements = list(map(float, line.split()[1::2]))
            if 'XX' in line:
               np.fill_diagonal(quad_matrix, m_elements)
            elif 'XY' in line:
               xs, ys = np.triu_indices(n_dim,k=1)
               quad_matrix[xs,ys] = m_elements
               quad_matrix[ys,xs] = m_elements
      # quad_matrix = np.expand_dims(quad_matrix, axis=0)
      return quad_matrix

   @property
   def get_mulliken(self):
      count = -1
      start_reading = False
      charges = list()
      pattern1 = 'Mulliken charges:'
      pattern2 = 'Mulliken atomic charges:'
      for line in self.lines:
         if pattern1 in line or pattern2 in line:
            start_reading = True
            continue
         if start_reading:
            count += 1
            if count >= 1:
               if len(line.split()) != 3:
                  break
               charges.append(line.split()[-1])
      charges = np.array(charges, dtype=np.float64)
      n_atoms = len(charges)
      # charges = charges.reshape(1, n_atoms,1)
      charges = charges.reshape(1,n_atoms)
      return charges

   @property
   def _get_exact_polar(self):
      for line in self.lines:
         if 'Exact polarizability' in line:
            polar = np.array(line.split()[-6:], dtype=np.float64)
            # polar = polar.reshape(-1,6)
            return polar

   @property
   def get_polarizability(self):
      start_reading = False
      scinot = re.compile('[-+]?[\d]+\.?[\d]*[Dd](?:[-+]?[\d]+)?')

      for line in self.lines:
         if 'Polarizability' in line:
            start_reading = True
            p1 = [x.replace('D','E') for x in re.findall(scinot, line)]
         if start_reading:
            p2 = [x.replace('D','E') for x in re.findall(scinot, line)]
            polar = np.array((p1 + p2), dtype=np.float64)
            polar = polar.reshape(-1,6)
            return polar

   @property
   def get_zpve(self):
      start_reading = False
      for line in self.lines:
         if 'vibrational energy' in line:
            start_reading = True
            continue
         if start_reading:
            zpve = np.array(line.split()[0], dtype=np.float64, ndmin=2)
            return zpve

   @property
   def get_rot_constants(self):
      for line in self.lines:
         if 'Rotational constants' in line:
            rot_const = np.array(line.split()[-3:], dtype=np.float64)
            rot_const = rot_const.reshape(1,3)
            return rot_const

   @property
   def get_elec_spatial_ext(self):
      for line in self.lines:
         if 'Electronic spatial extent' in line:
            r2 = np.array(line.split()[-1], dtype=np.float64, ndmin=2)
            return r2

   @property
   def get_thermal_energies(self):
      count = 0
      thermal_energies = list()
      pattern = 'Sum of electronic and thermal'
      for line in self.lines:
         if pattern in line:
            thermal_energies.append(line.split()[-1])
            count += 1
         if count == 3:
            break
      thermal_energies = np.array(thermal_energies, dtype=np.float64,)
      thermal_energies *= 627.5096080305927
      thermal_energies = thermal_energies.reshape(-1,3)
      return thermal_energies

   @property
   def get_frequencies(self):
      freq_list = list()
      pattern = 'Frequencies '
      for line in self.lines:
         if pattern in line:
            f = line.split()[2:]
            f += [np.NaN] * (3 - len(f))
            freq_list.append(f)
      freqs = np.array(freq_list, dtype=np.float64)

      # freqs = np.reshape(freqs, (-1,3))
      # freqs = freqs.flatten()
      # freqs = np.expand_dims(freqs, axis=0)
      return freqs
   @property
   def get_heat_capacity(self):
      found_cv_line=False
      for i, line in enumerate(self.lines):

         if 'CV' in line:
            found_cv_line = True

         elif found_cv_line and 'Cal/Mol-Kelvin' in line:
            next_line = self.lines[i + 1]
            values = next_line.split()
            extracted_value = values[1]
      cv = np.array(float(extracted_value), dtype=np.float64, ndmin=2)
      return cv
   
def main_import(log_file, out_file, work_type):
   g16_output = Gaussian16(log_file, work_type)
   if work_type =='opt':
      all_properties = {'labels': g16_output.get_atom_labels,
                        'coords': g16_output.get_coords,
                        'Etot': g16_output.get_total_energy,
                        'e_homo_lumo': g16_output.get_homo_lumo,
                        'dipole': g16_output.get_dipole,
                        'quadrupole': g16_output.get_quadrupole,
                        'rot_constants': g16_output.get_rot_constants,
                        'elec_spatial_ext': g16_output.get_elec_spatial_ext,
                        'forces': g16_output.get_forces,
                        'mulliken': g16_output.get_mulliken,
                        }
   elif work_type =='opt+freq':
   #opt+freq or single_point_calculation
      all_properties = {'labels': g16_output.get_atom_labels,
                        'coords': g16_output.get_coords,
                        'Etot': g16_output.get_total_energy,
                        'e_homo_lumo': g16_output.get_homo_lumo,
                        'polarizability': g16_output.get_polarizability,
                        'dipole': g16_output.get_dipole,
                        'quadrupole': g16_output.get_quadrupole,
                        'zpve': g16_output.get_zpve,
                        'rot_constants': g16_output.get_rot_constants,
                        'elec_spatial_ext': g16_output.get_elec_spatial_ext,
                        'forces': g16_output.get_forces,
                        'e_thermal': g16_output.get_thermal_energies,
                        'freqs': g16_output.get_frequencies,
                        'mulliken': g16_output.get_mulliken,
                        'cv': g16_output.get_heat_capacity,
                        }
   elif work_type == 'sp':
      all_properties = {'labels': g16_output.get_atom_labels,
                        'coords': g16_output.get_coords,
                        'Etot': g16_output.get_total_energy,
                        'e_homo_lumo': g16_output.get_homo_lumo,
                        'dipole': g16_output.get_dipole,
                        'quadrupole': g16_output.get_quadrupole,
                        'rot_constants': g16_output.get_rot_constants,
                        'elec_spatial_ext': g16_output.get_elec_spatial_ext,
                        'mulliken': g16_output.get_mulliken,
                        }
   return all_properties




if __name__ == '__main__':
   main_import()
