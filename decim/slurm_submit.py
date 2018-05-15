
#!/usr/bin/env python
'''
Parallelize functions with simple arguments via SLURM
'''

import subprocess
import tempfile
import os
import errno


def submit(walltime, memory, tmpdir, cwd, script, name,
           nodes=1, tasks=1,
           shellfname=None):
    '''
    Submit a script to torque
    '''

    sbatch_directives = '''
    #!/bin/env python
    #SBATCH --job-name={name}
    #SBATCH --nodes={nodes}
    #SBATCH --tasks-per-node={tasks}
    #SBATCH --time={walltime}
    #SBATCH --export=NONE
    #SBATCH --memory={memory}


    cd {cwd}
    mkdir -p cluster
    chmod a+rwx cluster
    #### set journal & error options
    #SBATCH -error {cwd}/cluster/$SLURM_JOB_ID.o
    #SBATCH -output {cwd}/cluster/$SLURM_JOB_ID.e
    '''.format(**{'walltime': walltime,
                  'nodes': nodes,
                  'memory': memory,
                  'tasks': tasks,
                  'name': name,
                  'cwd': cwd})

    environment_variables = '''
    module purge
    module load env
    module load site/slurm



    srun python3 {script}
    '''.format({'script': script})
    command = sbatch_directives + environment_variables
    with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir,
                                     prefix='delete_me_tmp') as shellfname:
        shellfname.write(command)
        shellfname = shellfname.name
    command = "sbatch %s" % (shellfname)
    output = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        shell=True)
    return output


def to_script(func, tmpdir, *args):
    '''
    Write a simple stub python function that calls this function.
    '''

    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=tmpdir,
                                     prefix='delete_me_tmp') as script:
        code = """
print('Parameters:', '%s', '%s')
from %s import %s
%s(*%s)
        """ % (str(args).replace("'", ''), func.__name__,
               func.__module__, func.__name__,
               func.__name__, str(args))
        script.write(code)
        return script.name


def pmap(func, args, walltime=12, memory=10, logdir=None, tmpdir=None,
         name=None, tasks=16, env=None, nodes=1):
    if name is None:
        name = func.__name__
    if logdir is None:
        from os.path import expanduser, join
        home = expanduser("~")
        logdir = join(home, 'cluster_logs', func.__name__)
        mkdir_p(logdir)
    if tmpdir is None:
        from os.path import expanduser, join
        home = expanduser("~")
        tmpdir = join(home, 'cluster_logs', 'tmp')
        mkdir_p(tmpdir)
    out = []
    script = to_script(func, tmpdir, *arg)
    print(script)
    pid = submit(walltime=walltime, memory=memory, cwd=logdir,
                 tmpdir=tmpdir, script=script, name=name, nodes=nodes,
                 tasks=tasks)
    out.append(pid)
    return out


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
