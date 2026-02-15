# =============================================================================
# File: tempdatasetfile.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

class SVFUltraMaskedSequenceDataset(Dataset):
    def __init__(self, opt, csv_file, svf_file, fold, subset='train', aug_p=0.8,
                 timepoints=['sc', 'm06', 'm12', 'm18', 'm24', 'm36'], target_timepoint='m18',
                 log_missing_scans=False, use_val_dataset=False):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[f'fold_{fold}'] == subset].reset_index(drop=True)
        self.opt = opt

        # Define MONAI transforms
        self.test_transforms = t.Compose([
            t.ToTensor(),
            NormalizeIntensity()
        ])
        self.monai_transforms = t.Compose([
            t.ToTensor(),
            t.Rand3DElastic(
                sigma_range=(0.01, 1),
                magnitude_range=(0, 1),
                prob=aug_p,
                rotate_range=(0.18, 0.18, 0.18),
                translate_range=(4, 4, 4),
                scale_range=(0.10, 0.10, 0.10),
                spatial_size=None,
                padding_mode="border"
            ),
            t.RandHistogramShift(num_control_points=(5, 15), prob=aug_p),
            t.RandAdjustContrast(prob=aug_p),
            t.RandGaussianNoise(prob=aug_p),
            t.NormalizeIntensity()
        ])

        if subset == 'train' and not use_val_dataset:
            val_df = pd.read_csv(csv_file)
            val_df = val_df[val_df[f'fold_{fold}'] == 'val'].reset_index(drop=True)
            self.df = pd.concat([self.df, val_df], ignore_index=True)

        self.df['Acq Date'] = pd.to_datetime(self.df['Acq Date'])
        self.df = self.df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')

        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        self.subset = subset
        self.aug_p = aug_p
        self.log_missing_scans = log_missing_scans
        
        # Set the segmentation masks directory
        self.mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_segs/"
        self.all_mask_files = os.listdir(self.mask_dir)

        # Grouping subjects
        self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
        self.subjects = list(self.subject_groups.index.levels[0])

        # Load SVF mapping: {subject -> {visit -> svf_path}}
        svf_df = pd.read_csv(svf_file)
        self.svf_map = {}
        for _, row in svf_df.iterrows():
            subj = row['Subject']
            visit = row['Visit']
            svf_path = row['SVF_Path']
            self.svf_map.setdefault(subj, {})[visit] = svf_path

        if self.target_timepoint == "rand":
            self.subject_target_timepoints = self._assign_random_target_timepoints()
        elif self.target_timepoint == "all":
            # For each subject, create multiple entries with different target timepoints
            self.subject_combinations = []
            for subject in self.subjects:
                # Get the actually available timepoints for this subject
                available_visits = set(self.df[self.df['Subject'] == subject]['Visit'].unique())
                valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]  # Skip 'sc'
                
                # Only include timepoints that have actual scans available
                for target_tp in valid_timepoints:
                    # Check if target timepoint scan actually exists
                    if target_tp in available_visits:
                        self.subject_combinations.append((subject, target_tp))
        else:
            self.subject_target_timepoints = {subject: self.target_timepoint for subject in self.subjects}
            subjects_with_target = self.df[self.df['Visit'] == self.target_timepoint]['Subject'].unique()
            self.df = self.df[self.df['Subject'].isin(subjects_with_target)].reset_index(drop=True)
            self.subject_groups = self.df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
            self.subjects = list(self.subject_groups.index.levels[0])

        # create a mapping between file path and id so we can return the id in the dataloader and reconvert it to filepath string
        self.file_path_to_id = {}
        for i, row in self.df.iterrows():
            file_path = row['File_Path']
            subject_id = row['Subject']
            visit = row['Visit']
            # generate an int uid
            uid = hash(file_path) % (10 ** 8)
            self.file_path_to_id[file_path] = uid

    def _assign_random_target_timepoints(self):
        subject_target_map = {}
        for subject in self.subjects:
            available_visits = self.subject_groups.loc[subject]['Visit'].unique()
            valid_timepoints = [tp for tp in self.timepoints[1:] if tp in available_visits]
            if valid_timepoints:
                subject_target_map[subject] = random.choice(valid_timepoints)
            else:
                raise ValueError(f"No valid timepoints for subject {subject}")
        return subject_target_map

    def __len__(self):
        if self.target_timepoint == "all":
            return len(self.subject_combinations)
        return len(self.subjects)

    def load_scan(self, subject_id, visit='sc'):
        visit_data = self.df[(self.df['Subject'] == subject_id) & (self.df['Visit'] == visit)]
        if visit_data.empty:
            raise ValueError(f"No {visit} scan found for subject {subject_id}")
        file_path = visit_data.iloc[0]['File_Path']
        nii = nib.load(file_path)
        scan = nii.get_fdata()
        if scan.ndim == 3:
            scan = np.expand_dims(scan, axis=0)  # add channel dimension

        # apply transform
        scan = self.test_transforms(scan)
        return scan
    
    def load_segmentation(self, image_id):
        """Load segmentation mask for a given image ID"""
        mask_files = [f for f in self.all_mask_files if image_id in f]
        seg_file = None
        for k in mask_files:
            if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in k:
                seg_file = k
                break
        
        if not seg_file:
            return torch.zeros((128, 128, 128)).float()
        
        mask_nib = nib.load(os.path.join(self.mask_dir, seg_file))
        mask_data = mask_nib.get_fdata()
        
        # Convert to tensor
        mask_tensor = torch.tensor(mask_data).float()
        return mask_tensor
    
    # def a function to get file_path mapping externally
    def get_file_path_mapping(self):
        uid_to_file_path = {}
        for file_path, uid in self.file_path_to_id.items():
            if uid not in uid_to_file_path:
                uid_to_file_path[uid] = file_path
        return uid_to_file_path

    def __getitem__(self, idx):
        if self.target_timepoint == "all":
            subject_id, target_timepoint = self.subject_combinations[idx]
        else:
            subject_id = self.subjects[idx]
            target_timepoint = self.subject_target_timepoints[subject_id]

        subject_data = self.subject_groups.loc[subject_id]
        
        mask = []

        # Load screening scan as latest_scan
        latest_scan = self.load_scan(subject_id, visit='sc')

        # Load target scan
        target_scan = self.load_scan(subject_id, visit=target_timepoint)
        target_scan_path = subject_data[subject_data['Visit'] == target_timepoint].iloc[0]['File_Path']
        target_uid = self.file_path_to_id[target_scan_path]
        
        # Get image IDs for segmentation loading
        target_image_id = subject_data[subject_data['Visit'] == target_timepoint].iloc[0]['Image Data ID']
        baseline_image_id = subject_data[subject_data['Visit'] == 'sc'].iloc[0]['Image Data ID']
        
        # Load segmentation masks
        target_seg = self.load_segmentation(target_image_id)
        baseline_seg = self.load_segmentation(baseline_image_id)

        # SVF stack for all timepoints except sc
        svfs = []
        # get the target SVF field
        target_svf_path = self.svf_map[subject_id][target_timepoint]
        target_svf = torch.load(target_svf_path).numpy()  # [3, H, W, D]

        target_timepoint_index = self.timepoints.index(target_timepoint) - 1
        for t, timepoint in enumerate(self.timepoints):
            if timepoint == 'sc':
                continue

            if timepoint == target_timepoint:
                # add mask
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))   
                continue

            # if not self.opt.use_future_data then skip scans after target_timepoint
            # if self.target_timepoint == "all" then add a mask instead of skipping
            if self.opt.use_future_data == False:
                if self.target_timepoint == "all":
                    if t > self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t > self.timepoints.index(target_timepoint):
                        continue

            if self.opt.use_past_data == False:
                if self.target_timepoint == "all":
                    if t < self.timepoints.index(target_timepoint):
                        mask.append(0)
                        svfs.append(np.zeros((3, 128, 128, 128)))
                        continue
                else:
                    if t < self.timepoints.index(target_timepoint):
                        continue
                
            visit_row = subject_data[subject_data['Visit'] == timepoint]
            if visit_row.empty:
                mask.append(0)
                svfs.append(np.zeros((3, 128, 128, 128)))
            else:
                mask.append(1)
                try:
                    svf_path = self.svf_map[subject_id][timepoint]
                    svf = torch.load(svf_path).numpy()  # [3, H, W, D]
                except KeyError:
                    svf = np.zeros((3, 128, 128, 128))
                svfs.append(svf)
                if svf.shape != (3, 128, 128, 128):
                    raise ValueError(f"SVF shape mismatch for subject {subject_id}, timepoint {timepoint}: {svf.shape}")
                
        stacked_svf = torch.tensor(np.stack(svfs)).float()
        latest_scan = latest_scan.float()
        target_scan = target_scan.float()
        mask = torch.tensor(mask, dtype=torch.float32)

        group_map = {'CN': 0, 'MCI': 1, 'AD': 2}
        timepoint_map = {tp: i for i, tp in enumerate(self.timepoints)}

        class_label = torch.tensor(group_map[subject_data.iloc[0]['Group']]).unsqueeze(0)
        reg_label = torch.tensor(subject_data.iloc[0]['Age']).unsqueeze(0)
        target_timepoint_tensor = torch.tensor(timepoint_map[target_timepoint]).unsqueeze(0)
        latest_scan_timepoint = torch.tensor(timepoint_map['sc']).unsqueeze(0)
        target_timepoint_index = torch.tensor(target_timepoint_index).unsqueeze(0)
        target_svf = torch.tensor(target_svf).float()
        target_uid = torch.tensor(target_uid).unsqueeze(0).int()

        return {
            "stacked_svf": stacked_svf,                       # [T-1, 3, H, W, D]
            "latest_scan": latest_scan,                       # [1, H, W, D]
            "target_scan": target_scan,                       # [1, H, W, D]
            "class_label": class_label,                       # [1]
            "reg_label": reg_label,                           # [1]
            "target_timepoint": target_timepoint_tensor,      # [1]
            "latest_timepoint": latest_scan_timepoint,        # [1]
            "mask": mask,                                     # [T]
            "target_timepoint_index": target_timepoint_index, # [1]
            "target_svf": target_svf,                         # [3, H, W, D]
            "target_uid": target_uid,                         # [1]
            "target_seg": target_seg,                         # [H, W, D] 
            "baseline_seg": baseline_seg                      # [H, W, D]
        }