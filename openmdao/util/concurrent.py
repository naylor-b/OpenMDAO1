
from __future__ import print_function

import os
import sys
import traceback
from six import next

from openmdao.core.mpi_wrap import under_mpirun, MPI, debug

trace = os.environ.get("OPENMDAO_TRACE")

def concurrent_eval_lb(func, cases, comm, broadcast=False):
    """
    Runs a load balanced version of the function, with the master
    rank (0) sending a new case to each worker rank as soon as it
    has finished its last case.

    Args
    ----

    func : function
        The function to execute in workers.

    cases : collection of function args
        Queue entries are assumed to be of the form (args,) or (args, kwargs).

    com : MPI communicator
        The MPI communicator that is shared between the master and workers.

    broadcast : bool, optional
        If True, the results will be broadcast out to the worker procs so
        that the return value of concurrent_eval_lb will be the full result
        list in every process.
    """
    if MPI:
        if comm.rank == 0:  # master rank
            results = _concurrent_eval_lb_master(cases, comm)
        else:
            results = _concurrent_eval_lb_worker(func, comm)

        if broadcast:
            results = comm.bcast(results, root=0)

    else: # serial execution
        results = []
        for args, kwargs in cases:
            try:
                if kwargs:
                    retval = func(*args, **kwargs)
                else:
                    retval = func(*args)
            except:
                err = traceback.format_exc()
                retval = None
            else:
                err = None
            results.append((retval, err))

    return results

def _concurrent_eval_lb_master(cases, comm):
    received = 0
    sent = 0

    results = []

    case_iter = iter(cases)

    # seed the workers
    for i in range(1, comm.size):
        try:
            case = next(case_iter)
        except StopIteration:
            break

        if trace: # pragma: no cover
            debug('Sending Seed case %d' % i)

        comm.send(case, i, tag=1)
        sent += 1

        if trace: # pragma: no cover
            debug('Seed Case Sent %d' % i)

    # send the rest of the cases
    if sent > 0:
        while True:
            if trace: # pragma: no cover
                debug("Waiting on case")

            worker, retval, err = comm.recv(tag=2)

            if trace:  # pragma: no cover
                debug("Case Recieved from Worker %d" % worker )

            received += 1

            # store results
            results.append((retval, err))

            # don't stop until we hear back from every worker process
            # we sent a case to
            if received == sent:
                break

            try:
                case = next(case_iter)
            except StopIteration:
                pass
            else:
                # send new case to the last worker that finished
                if trace: # pragma: no cover
                    debug("Sending New Case to Worker %d" % worker )
                comm.send(case, worker, tag=1)
                sent += 1
                if trace: # pragma: no cover
                    debug("Case Sent to Worker %d" % worker )

    # tell all workers to stop
    for rank in range(1, comm.size):
        if trace: # pragma: no cover
            debug("Make Worker Stop on Rank %d" % rank )
        comm.send((None, None), rank, tag=1)
        if trace: # pragma: no cover
            debug("Worker has Stopped on Rank %d" % rank )

    return results

def _concurrent_eval_lb_worker(func, comm):
    while True:
        # wait on a case from the master
        if trace: debug("Receiving Case from Master") # pragma: no cover

        args, kwargs = comm.recv(source=0, tag=1)

        if trace: debug("Case Received from Master") # pragma: no cover

        if args is None: # we're done
            if trace: debug("Received None, quitting") # pragma: no cover
            break

        try:
            if kwargs:
                retval = func(*args, **kwargs)
            else:
                retval = func(*args)
        except:
            err = traceback.format_exc()
            retval = None
        else:
            err = None

        # tell the master we're done with that case
        if trace: debug("Send Master Return Value") # pragma: no cover

        comm.send((comm.rank, retval, err), 0, tag=2)

        if trace: debug("Return Value Sent to Master") # pragma: no cover


if __name__ == '__main__':
    def funct(job, option=None):
        if job == 5:
            raise RuntimeError("Job 5 had an error!")
        print("Running job %d" % job)
        if MPI:
            rank = MPI.COMM_WORLD.rank
        else:
            rank = 0
        return (job, option, rank)

    if MPI:
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        comm = None
        rank = 0

    if len(sys.argv) > 1:
        ncases = int(sys.argv[1])
    else:
        ncases = 10

    cases = [([i], {'option': 'foo%d'%i}) for i in range(ncases)]

    results = concurrent_eval_lb(funct, cases, comm)

    if MPI is None or comm.rank == 0:
        print("Results:")
        for r in results:
            print(r)

    if MPI:
        comm.barrier()

    print("------ broadcast ----")
    if ncases > 5:
        cases.remove(([5], {'option': 'foo5'}))  # git rid of excetption case
    results = concurrent_eval_lb(funct, cases, comm, broadcast=True)

    print("Results for rank %d: %s" % (rank, results))
