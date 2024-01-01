
import re
import numpy as np
from parse_g16.src.sub_code_get_GS_opt_props import Gaussian16

class Gaussian16_ES(Gaussian16):
   def __init__(self, output, work_type):
      super().__init__(output, work_type)
      self.output=output
      with open(output, 'r') as file:
            lines = file.readlines()
      if work_type == 'opt':
         new_lines = []
         for line in reversed(lines):
               new_lines.append(line)
               if 'Excited states from' in line:
                  break
         new_lines.reverse()
         self.lines = new_lines


   @property
   def get_total_energy(self):
      with open(self.output, 'r') as f1:
         for line in f1:
            if 'SCF Done' in line:
               total_energy = np.array(line.split()[4], dtype=np.float64, ndmin=2)
               # output energy in kcal/mol
               total_energy *= 627.5096080305927
               break
      return total_energy

   def _get_transition_DMs(self, dm_type):
      finding=False
      dms = []
      for line in self.lines:
         if dm_type in line:
            finding = True
            continue
         if finding == True and line.split()[0]!='state':
            if not line.lstrip()[0].isdigit():
               break
            dms.append(list(map(lambda x: round(float(x), 4), line.split()[1:])))
      dm = np.vstack(dms)
      return dm

   @property
   def get_transition_electric_DM(self):
      dm_type = 'Ground to excited state transition electric dipole moments'
      dm = self._get_transition_DMs(dm_type)
      return dm

   @property
   def get_transition_velocity_DM(self):
      dm_type = 'Ground to excited state transition velocity dipole moments'
      dm = self._get_transition_DMs(dm_type)
      return dm

   @property
   def get_transition_magnetic_DM(self):
      dm_type = 'Ground to excited state transition magnetic dipole moments'
      dm = self._get_transition_DMs(dm_type)
      return dm

   @property
   def get_transition_velocity_quadrupole_moments(self):
      dm_type = ' Ground to excited state transition velocity quadrupole moments'
      dm = self._get_transition_DMs(dm_type)
      return dm

   @property
   def count_OrbNum_HomoLumo(self):
      occupied_count = 0
      # virtual_count = 0
      current_section = None
      for line in self.lines:
         if "Occupied" in line:
            current_section = "Occupied"
         elif "Virtual" in line:
            current_section = "Virtual"

         matches = re.findall(r'\(.*?\)', line)
         if current_section == "Occupied":
            occupied_count += len(matches)

            # occupied_count += line.count("(A)") if current_section == "Occupied" else 0
            # virtual_count += line.count("(A)") if current_section == "Virtual" else 0
         elif current_section == "Virtual":
            break

      return [str(occupied_count), str(occupied_count+1)]


   @property
   def get_Info_FromAllExcitedStates(self):
      all_dict = {}
      state_dict = {}
      orbs = {}

      for line in self.lines:
         if ' Excited State   ' in line:
               state_dict = {
                  'state': int(line.split()[2].replace(':', '')),
                  'state_type': line.split('-')[0].split()[-1],
                  'oscillator_trength': float(line.split('=')[1].split()[0]),
                  'excitation_e_eV': ' '.join(line.split()[4:6]),
                  'excitation_e_nm': ' '.join(line.split()[6:8])
               }
               orbs = {}
         elif (state_dict and line.split()) and (state_dict and line.split()[0].isdigit()):
               if '->' in line:
                  orb_i = line.split('->')[0].strip() + '->' + line.split('->')[1].split()[0]
               elif '<-' in line:
                  orb_i = line.split('<-')[1].split()[0] + '->' + line.split('<-')[0].strip()
               orbs[orb_i] = float(line.split()[-1])
         elif (state_dict and not line.split()) or (state_dict and not line.split()[0].isdigit()):
               top_three = sorted(orbs.items(), key=lambda x: abs(x[1]), reverse=True)
               state_dict['topExcitationContribution'] = top_three[:3]
               all_dict[state_dict['state']] = state_dict
               state_dict = {}
         if 'SavETr' in line:
               break
      return all_dict


def main_import(log_file, out_file, work_type):
   g16_output = Gaussian16_ES(log_file, work_type)
   if work_type == 'opt+freq':
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
                        'transition_electric_DM':g16_output.get_transition_electric_DM,
                        'transition_velocity_DM':g16_output.get_transition_velocity_DM,
                        'transition_magnetic_DM':g16_output.get_transition_magnetic_DM,
                        'transition_velocity_quadrupole_moments':g16_output.get_transition_velocity_quadrupole_moments,
                        'OrbNum_HomoLumo':g16_output.count_OrbNum_HomoLumo,
                        'Info_of_AllExcitedStates':g16_output.get_Info_FromAllExcitedStates,
                        }
   elif work_type == 'opt':
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
                        'transition_electric_DM':g16_output.get_transition_electric_DM,
                        'transition_velocity_DM':g16_output.get_transition_velocity_DM,
                        'transition_magnetic_DM':g16_output.get_transition_magnetic_DM,
                        'transition_velocity_quadrupole_moments':g16_output.get_transition_velocity_quadrupole_moments,
                        'OrbNum_HomoLumo':g16_output.count_OrbNum_HomoLumo,
                        'Info_of_AllExcitedStates':g16_output.get_Info_FromAllExcitedStates,
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
                        'transition_electric_DM':g16_output.get_transition_electric_DM,
                        'transition_velocity_DM':g16_output.get_transition_velocity_DM,
                        'transition_magnetic_DM':g16_output.get_transition_magnetic_DM,
                        'transition_velocity_quadrupole_moments':g16_output.get_transition_velocity_quadrupole_moments,
                        'OrbNum_HomoLumo':g16_output.count_OrbNum_HomoLumo,
                        'Info_of_AllExcitedStates':g16_output.get_Info_FromAllExcitedStates,
                        }
   return all_properties

if __name__ == '__main__':
   # main()
   main_import('/data/home/zhuyf/dataset_work/database/qm9_database/Gaussian_gau_excited_wb97xd/11/11.log','./aaa.dat','sp')
   #main_xyz('/data/home/zhuyf/dataset_work/database/qm9_database/Gaussian/777/777.log','./aaa.xyz')

