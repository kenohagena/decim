
#!/usr/bin/env python
'''
Parallelize functions with simple arguments via SLURM
'''

import subprocess
import tempfile
import os
import errno


def submit(walltime, memory, tmpdir, cwd, script, name,
           nodes=1, tasks=16,
           shellfname=None):
    '''
    Submit a script to torque
    '''
    print('script in submit {}'.format(script))
    sbatch_directives = '''#!/bin/sh
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={tasks}
#SBATCH --time={walltime}
#SBATCH --export=NONE
#SBATCH --memory={memory}
#SBATCH --partition=std


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

cd tmpdir

srun python3 {script}

cp -r tmpdir /work/faty014
    '''.format(**{'script': script})
    command = sbatch_directives + environment_variables
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=tmpdir,
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
from {module} import {function}
{function}{args}
        """.format(**{'module': func.__module__,
                      'function': func.__name__,
                      'args': args})
        script.write(code)
        return str(script.name)


def pmap(func, *args, walltime=12, memory=10, tmpdir=None,
         name=None, tasks=16, env=None, nodes=1):
    if name is None:
        name = func.__name__
    if tmpdir is None:
        from os.path import join
        tmp=tempfile.TemporaryDirectory().name
        tmpdir = join(tmp, 'cluster_logs', 'tmp')
        logdir = join(tmp, 'cluster_logs', func.__name__)
        mkdir_p(tmpdir)
        mkdir_p(logdir)
    out = []
    script = to_script(func, tmpdir, *args)
    print('script in pmap = {}'.format(script))
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
