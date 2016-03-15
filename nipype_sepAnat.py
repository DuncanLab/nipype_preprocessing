import os                                   # system functions
import nipype.interfaces.fsl as fsl         # fsl
import nipype.interfaces.afni as afni       # afni
import nipype.interfaces.ants as ants       # ants
import nipype.pipeline.engine as pe         # the workflow and node wrappers
import nipype.interfaces.io as nio          # Input/Output
import nipype.interfaces.utility as util    # utility

#======================================================================
# 1. Variable Specification
#======================================================================

experiment_dir = '/Volumes/homes/Shafquat/'             # location of data folder

# Count all the subfolders within a given directory
subs = next(os.walk(experiment_dir+'/Subjects/'))[1]
subject_list = [] # Initialize an empty list to store subjects
session_list = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5']              # list of session identifiers
# Set a last run based on the list of runs
last_run = session_list[-1]                 # Make sure to change the hardcoded last_run value within the picklast function below

# list of subject identifiers
for subject in subs:
    subject_list.append(subject)

output_dir = 'output_firstSteps'          # name of output folder
working_dir = 'workingdir_firstSteps'     # name of working directory

number_of_slices = 40                     # number of slices in volume
TR = 2.0                                  # time repetition of volume
smoothing_size = 6                        # size of FWHM in mm
number_volumes_trim = 8                   # size of FWHM in mm
#======================================================================
# 2. NODE SPECIFICATION
#======================================================================

volTrim = pe.Node(fsl.ExtractROI(t_min = number_volumes_trim,
                                 t_size=-1),
                  name='volTrim')

sliceTiming = pe.Node(fsl.SliceTimer(time_repetition = TR,
                                     interleaved = True),
                   name="sliceTiming")

motionCorrection = pe.Node(afni.Volreg(md1d_file = 'max_disp.1d',
                                       oned_file = 'mot_par.1d',
                                       zpad = 3,
                                       outputtype = 'NIFTI_GZ'),
                                       name="motionCorrection")

susan = pe.Node(fsl.SUSAN(brightness_threshold = 150,
                               fwhm = 3),
                     name="susan")

skullStrip = pe.Node(afni.SkullStrip(args = '-niter 400 -push_to_edge -fill_hole -touchup -touchup',
                    outputtype = 'NIFTI_GZ'),
                    name="skullStrip")

betFunc = pe.Node(fsl.BET(frac = 0.5,
                    vertical_gradient = 0),
                    name="betFunc")

tstat = pe.Node(afni.TStat(args = '-mean',
                           outputtype = 'NIFTI_GZ'),
                name="tstat")


mean2anatAnts = pe.Node(ants.Registration(args='--float',
                                            metric=['MI'],
                                            metric_weight=[1.0],
                                            shrink_factors=[[8,4,2,1]],
                                            smoothing_sigmas=[[3.0,2.0,1.0,0.0]],
                                            transforms=['Rigid'],
                                            transform_parameters=[(0.1,)],
                                            number_of_iterations=[[1000,500,250,100]],
                                            write_composite_transform = True,
                                            convergence_threshold=[1e-06],
                                            convergence_window_size=[10],
                                            output_warped_image='output_warped_image.nii.gz'),
                                            name='mean2anatAnts')


anat2MNI = pe.Node(ants.Registration(args='--float',
                                            metric=['Mattes'] * 2 + [['Mattes', 'CC']],
                                            metric_weight=[1] * 2 + [[0.5, 0.5]],
                                            radius_or_number_of_bins = [32] * 2 + [[32, 4]],
                                            sampling_strategy = ['Regular'] * 2 + [[None, None]],
                                            sampling_percentage = [0.3] * 2 + [[None, None]],
                                            use_histogram_matching = [False] * 2 + [True],
                                            shrink_factors=[[3, 2, 1]] * 2 + [[4, 2, 1]],
                                            smoothing_sigmas=[[4, 2, 1]] * 2 + [[1, 0.5, 0]],
                                            sigma_units = ['vox'] * 3,
                                            transforms=['Rigid', 'Affine', 'SyN'],
                                            transform_parameters=[(0.1,),(0.1,), (0.2, 3.0, 0.0)],
                                            number_of_iterations=[[10000, 11110, 11110]] * 2 + [[100, 30, 20]],
                                            write_composite_transform = True,
                                            collapse_output_transforms = True,
                                            initial_moving_transform_com = True,
                                            convergence_threshold= [1.e-8] * 2 + [-0.01],
                                            convergence_window_size=[20] * 2 + [5],
                                            use_estimate_learning_rate_once = [True] * 3,
                                            winsorize_lower_quantile = 0.005,
                                            winsorize_upper_quantile = 0.995,
                                            num_threads = 2,
                                            #output_transform_prefix = "MNI_warped_",
                                            output_warped_image='MNI_warped_image.nii.gz'),
                                            name='anat2MNI')
anat2MNI.plugin_args = {'qsub_args': '-pe orte 4',
                       'sbatch_args': '--mem=6G -c 4'}



merge = pe.Node(util.Merge(2), iterfield=['in2'], name='mergexfm')

warpmean = pe.Node(ants.ApplyTransforms( input_image_type = 0,
                                         interpolation = 'Linear',
                                         invert_transform_flags = [False, False],
                                         terminal_output = 'file'),
                                         name='warpmean')


applyTransFunc = pe.Node(ants.ApplyTransforms(input_image_type = 3,
                                        interpolation = 'BSpline',
                                        invert_transform_flags = [False, False],
                                        terminal_output = 'file'),
                                        iterfield=['input_image', 'transforms'],
                                        name='applyTransFunc')

#======================================================================
# 3. SPECIFY WORKFLOWS
#======================================================================
# Functional Workflow
preproc = pe.Workflow(name='preproc')
preproc.base_dir = os.path.join(experiment_dir, working_dir)

# Anatomical Workflow
preprocAnat = pe.Workflow(name='anatpreproc')
preprocAnat.base_dir = os.path.join(experiment_dir, working_dir)


#======================================================================
# 3. INPUT/OUTPUT SPECIFICATION (func)
#======================================================================

# Infosource - a function free node to iterate over the list of subject names
infosource = pe.Node(util.IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list)]

# SelectFiles for Input
templates = {'anat': experiment_dir + '/Subjects/{subject_id}/Anatomical/*.nii.gz',
             'func': experiment_dir + '/Subjects/{subject_id}/{session_id}/{session_id}.nii.gz'}


selectfiles = pe.Node(nio.SelectFiles(templates), name="selectfiles")

# DataSink for Output
datasink = pe.Node(nio.DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")


# Use the following DataSink output substitutions
substitutions = [('_subject_id', ''),
                 ('_session_id_', '')]
datasink.inputs.substitutions = substitutions


#======================================================================
# 3. INPUT/OUTPUT SPECIFICATION (anat)
#======================================================================

# Infosource - a function free node to iterate over the list of subject names
infosourceAnat = pe.Node(util.IdentityInterface(fields=['subject_id']),
                  name="infosource")

infosourceAnat.iterables = [('subject_id', subject_list)]

# SelectFiles for Input
templatesAnat = {'anat': experiment_dir + '/Subjects/{subject_id}/Anatomical/*.nii.gz'}


selectfilesAnat = pe.Node(nio.SelectFiles(templatesAnat), name="selectfilesAnat")

# DataSink for Anatomical Output
datasinkAnat = pe.Node(nio.DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasinkAnat")


# Use the following DataSink output substitutions
substitutions = [('_subject_id_', '')]
datasinkAnat.inputs.substitutions = substitutions


#======================================================================
# 4. Motion Correction
#======================================================================

# Set up an extract node to extract the last volume from the last run
extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1),
                      name = 'extractref')

# Pick the last file from the list of files
def  picklast(path_to_run):
    import os
    last_run = 'Run5'

    # get path to subject
    subject = os.path.dirname(os.path.dirname(path_to_run))
    selected_run = subject + "/" + last_run + "/" + last_run + ".nii.gz"

    return selected_run


preproc.connect(selectfiles, ('func', picklast), extract_ref, 'in_file')


# Pick the last volume from the given run
def getlastvolume(func):
    from nibabel import load
    funcfile = func
    _,_,_,timepoints = load(funcfile).get_shape()
    # To return middle volume use (timepoints/2)-1
    return (timepoints-1)

preproc.connect(sliceTiming, ('slice_time_corrected_file', getlastvolume), extract_ref, 't_min')

# Take the extracted last volume from the last run and use it as a reference file
preproc.connect([(sliceTiming, motionCorrection, [('slice_time_corrected_file', 'in_file')]),
                 (extract_ref, motionCorrection, [('roi_file', 'basefile')])
                 ])

#======================================================================
# 5. Remove skull from anatomical image and automask the functional image
#======================================================================



# Connect the mean last run with skull stripping
# preproc.connect([(tstat, func_skullStrip, [('out_file','in_file')])])

preprocAnat.connect([(infosourceAnat, selectfilesAnat, [('subject_id', 'subject_id')]),
                     (selectfilesAnat, susan, [('anat', 'in_file')]),
                     (susan, skullStrip, [('smoothed_file', 'in_file')]),
                    (skullStrip, datasinkAnat, [('out_file', 'skull_stripped')])])

#======================================================================
# 6. Connecting Workflows to datasink
#======================================================================

# Connect all components of the preprocessing workflow
# Connect SelectFiles and DataSink to the workflow
preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                            ('session_id', 'session_id')]),
                 (selectfiles, volTrim, [('func', 'in_file')]),
                 (volTrim, sliceTiming, [('roi_file', 'in_file')]),
                 (volTrim, datasink, [('roi_file', 'trimmed')]),
                 (sliceTiming, datasink, [('slice_time_corrected_file', 'sliced')]),
                 (motionCorrection, datasink, [('out_file', 'motion_correct')]),
                 (motionCorrection, datasink, [('md1d_file', 'mc_md1d')]),
                 (motionCorrection, datasink, [('oned_file', 'mc_oned')]),
                                  ])


#======================================================================
# 6.5 New preprocess pipeline to retrieve motion corrected files
#======================================================================

# Workflow after motion correction
preproc2 = pe.Workflow(name='preproc2')
preproc2.base_dir = os.path.join(experiment_dir, working_dir)

# Infosource - a function free node to iterate over the list of subject names
infosource2 = pe.Node(util.IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosource2")

infosource2.iterables = [('subject_id', subject_list)]

# Select the last run for each subject
templates2 = {'func2': experiment_dir + 'output_firstSteps/motion_correct/'+ last_run + '_{subject_id}/*.nii.gz',
              'noskull': experiment_dir + 'output_firstSteps/skull_stripped/{subject_id}/*.nii.gz',
              'MNI': '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'}

selectfiles2 = pe.Node(nio.SelectFiles(templates2), name="selectfiles2")

# DataSink
datasink2 = pe.Node(nio.DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink2")

# Use the following DataSink output substitutions
substitutions = [('_subject_id_', '')]
datasink2.inputs.substitutions = substitutions


# Set up a extract node for the last run after motion control and get the mean
preproc2.connect([(infosource2, selectfiles2, [('subject_id', 'subject_id')]),
                  (selectfiles2, tstat, [('func2', 'in_file')]),
                  (tstat, betFunc, [('out_file', 'in_file')]),
                  ])

preproc2.connect([(tstat, datasink2, [('out_file', 'tstat')]),
                  (betFunc, datasink2, [('out_file', 'betFunc')]),
                   ])

#======================================================================
# 7. Connect Anatomical to Mean Functional image
#======================================================================

preproc2.connect([(selectfiles2, mean2anatAnts, [('noskull', 'fixed_image')]),
                  (betFunc, mean2anatAnts, [('out_file', 'moving_image')]),
                  (mean2anatAnts, datasink2, [('warped_image', 'mean2anat')]),
                  #(mean2anatAnts, datasink2, [('forward_transforms', 'mean2anatMatrix')]),
                  (mean2anatAnts, datasink2, [('composite_transform', 'mean2anatMatrix_Composites')])
                   ])

#======================================================================
# 8. Compute registration between subjects' structural and MNI template
#======================================================================

preproc2.connect([(selectfiles2, anat2MNI, [('MNI', 'fixed_image')]),
                  (selectfiles2, anat2MNI, [('noskull', 'moving_image')]),
                  (anat2MNI, datasink2, [('warped_image', 'MNI_warped')]),
                  (anat2MNI, datasink2, [('composite_transform', 'MNI_warpedMatrix')])
                   ])

#======================================================================
# 8. register functional images anatomical and MNI template using ANTS in a new workflow
#======================================================================

# Registration Workflow
preprocReg = pe.Workflow(name='preprocRegister')
preprocReg.base_dir = os.path.join(experiment_dir, working_dir)

# Infosource - a function free node to iterate over the list of subject names
infosourceReg = pe.Node(util.IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosourceReg")

infosourceReg.iterables = [('subject_id', subject_list)]

# Select the last run for each subject
templatesReg = {'mean2anat': experiment_dir + 'output_firstSteps/mean2anat/{subject_id}/*.nii.gz',
                'mean2anatMatrix': experiment_dir + 'output_firstSteps/mean2anatMatrix_Composites/{subject_id}/*.h5',
                'MNI_warped': experiment_dir + 'output_firstSteps/MNI_warped/{subject_id}/*.nii.gz',
                'MNI_warpedMatrix': experiment_dir + 'output_firstSteps/MNI_warpedMatrix/{subject_id}/*.h5',
                'MNI': '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
                'func_mc': experiment_dir + 'output_firstSteps/motion_correct/{session_id}_{subject_id}/*.nii.gz'}

selectfilesReg = pe.Node(nio.SelectFiles(templatesReg), name="selectfilesReg")

# DataSink
datasinkReg = pe.Node(nio.DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasinkReg")

# Use the following DataSink output substitutions
substitutions = [('_subject_id', ''),
                 ('_session_id_', '')]
datasinkReg.inputs.substitutions = substitutions

preprocReg.connect([(infosource, selectfilesReg, [('subject_id', 'subject_id'),
                                            ('session_id', 'session_id')]),
                    (selectfilesReg, merge, [('MNI_warpedMatrix', 'in2')]),
                    (selectfilesReg, merge, [('mean2anatMatrix', 'in1')]),
                    (merge, applyTransFunc, [('out', 'transforms')]),
                    (selectfilesReg, applyTransFunc, [('func_mc', 'input_image')]),
                    (selectfilesReg, applyTransFunc, [('MNI', 'reference_image')]),
                    (applyTransFunc, datasinkReg, [('output_image', 'warpedfunc')]),
                   ])


#======================================================================
# 9. Run, Forrest, Run!
#======================================================================

# Run the Nodes
#preprocAnat.run('MultiProc', plugin_args={'n_procs': 3})
#preproc.run('MultiProc', plugin_args={'n_procs': 3})
#preproc.run('MultiProc', plugin_args={'n_procs': 3})
preprocReg.run('MultiProc', plugin_args={'n_procs': 1})

#======================================================================
# . Return output to subject directories
#======================================================================
"""
# get all the subfolders within a given directory
num_folders = next(os.walk(experiment_dir+'/output_firstSteps/final_output'))[1]

# move file
for folder in num_folders:
    dir_files = os.listdir(experiment_dir+'/output_firstSteps/final_output/'+folder+'/')
    for file_name in dir_files:
        full_file_name = experiment_dir+'/output_firstSteps/final_output/'+folder+'/' + '/' + file_name
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, experiment_dir+'/Subjects/'+folder[-6:]+'/'+folder[0:4]+'/')
"""
