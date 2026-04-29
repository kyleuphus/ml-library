kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
zsh: command not found: pytest
kyleuphus@Kyles-MacBook-Pro-4 ml-library % activatevenv
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py ...FF                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

========================================================================== FAILURES ===========================================================================
__________________________________________________________________________ test_tanh __________________________________________________________________________

    def test_tanh():
        a = Value(5.2)
        b = Value(0.5)
        c = (a * b).tanh()
        c.backward()
    
        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(-3.0, requires_grad=True)
        c_t = (a_t * b_t).tanh()
        c_t.backward()
    
>       assert a.grad == a_t.grad.item()
E       assert 0.010912398847672589 == -7.367088983301073e-05
E        +  where 0.010912398847672589 = Value(data=5.2).grad
E        +  and   -7.367088983301073e-05 = <built-in method item of Tensor object at 0x10f7b18b0>()
E        +    where <built-in method item of Tensor object at 0x10f7b18b0> = tensor(-7.3671e-05).item
E        +      where tensor(-7.3671e-05) = tensor(2., requires_grad=True).grad

tests/gradient_test.py:63: AssertionError
______________________________________________________________________ test_MLP_forward _______________________________________________________________________

    def test_MLP_forward():
        xs = [[2.0, 4.0], [1.5, 3.0]]
        ys = [1, 1]
    
        model = MLP(2, [1])
    
        model_p = nn.Linear(2, 1)
    
        weights = torch.tensor([weight.data for weight in model.layers[0].neurons[0].w])
        biases = torch.tensor([model.layers[0].neurons[0].b.data])
        with torch.no_grad():
            model_p.weight.copy_(weights)
            model_p.bias.copy_(biases)
    
        y_pred = model(xs[0])[0]
        y_pred_p = model_p(torch.tensor(xs[0]))
    
>       assert y_pred.data == y_pred_p.tensor.item()
                              ^^^^^^^^^^^^^^^
E       AttributeError: 'Tensor' object has no attribute 'tensor'

tests/gradient_test.py:84: AttributeError
=================================================================== short test summary info ===================================================================
FAILED tests/gradient_test.py::test_tanh - assert 0.010912398847672589 == -7.367088983301073e-05
FAILED tests/gradient_test.py::test_MLP_forward - AttributeError: 'Tensor' object has no attribute 'tensor'
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py ...FF                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

========================================================================== FAILURES ===========================================================================
__________________________________________________________________________ test_tanh __________________________________________________________________________

    def test_tanh():
        a = Value(5.2)
        b = Value(0.5)
        c = (a * b).tanh()
        c.backward()
    
        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(-3.0, requires_grad=True)
        c_t = (a_t * b_t).tanh()
        c_t.backward()
    
>       assert a.grad == a_t.grad.item()
E       assert 0.010912398847672589 == -7.367088983301073e-05
E        +  where 0.010912398847672589 = Value(data=5.2).grad
E        +  and   -7.367088983301073e-05 = <built-in method item of Tensor object at 0x10da259a0>()
E        +    where <built-in method item of Tensor object at 0x10da259a0> = tensor(-7.3671e-05).item
E        +      where tensor(-7.3671e-05) = tensor(2., requires_grad=True).grad

tests/gradient_test.py:63: AssertionError
______________________________________________________________________ test_MLP_forward _______________________________________________________________________

    def test_MLP_forward():
        xs = [[2.0, 4.0], [1.5, 3.0]]
        ys = [1, 1]
    
        model = MLP(2, [1])
    
        model_p = nn.Linear(2, 1)
    
        weights = torch.tensor([weight.data for weight in model.layers[0].neurons[0].w])
        biases = torch.tensor([model.layers[0].neurons[0].b.data])
        with torch.no_grad():
            model_p.weight.copy_(weights)
            model_p.bias.copy_(biases)
    
        y_pred = model(xs[0])[0]
        y_pred_p = model_p(torch.tensor(xs[0]))
    
>       assert y_pred.data == y_pred_p.item()
E       assert 2.673464388305629 == 2.673464298248291
E        +  where 2.673464388305629 = Value(data=2.673464388305629).data
E        +  and   2.673464298248291 = <built-in method item of Tensor object at 0x10da26ee0>()
E        +    where <built-in method item of Tensor object at 0x10da26ee0> = tensor([2.6735], grad_fn=<ViewBackward0>).item

tests/gradient_test.py:84: AssertionError
=================================================================== short test summary info ===================================================================
FAILED tests/gradient_test.py::test_tanh - assert 0.010912398847672589 == -7.367088983301073e-05
FAILED tests/gradient_test.py::test_MLP_forward - assert 2.673464388305629 == 2.673464298248291
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py ...FF                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

========================================================================== FAILURES ===========================================================================
__________________________________________________________________________ test_tanh __________________________________________________________________________

    def test_tanh():
        a = Value(5.2)
        b = Value(0.5)
        c = (a * b).tanh()
        c.backward()
    
        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(-3.0, requires_grad=True)
        c_t = (a_t * b_t).tanh()
        c_t.backward()
    
>       assert a.grad == a_t.grad.item()
E       assert 0.010912398847672589 == -7.367088983301073e-05
E        +  where 0.010912398847672589 = Value(data=5.2).grad
E        +  and   -7.367088983301073e-05 = <built-in method item of Tensor object at 0x10dea1a90>()
E        +    where <built-in method item of Tensor object at 0x10dea1a90> = tensor(-7.3671e-05).item
E        +      where tensor(-7.3671e-05) = tensor(2., requires_grad=True).grad

tests/gradient_test.py:64: AssertionError
______________________________________________________________________ test_MLP_forward _______________________________________________________________________

    def test_MLP_forward():
        xs = [[2.0, 4.0], [1.5, 3.0]]
        ys = [1, 1]
    
        model = MLP(2, [1])
    
        model_p = nn.Linear(2, 1)
    
        weights = torch.tensor([weight.data for weight in model.layers[0].neurons[0].w])
        biases = torch.tensor([model.layers[0].neurons[0].b.data])
        with torch.no_grad():
            model_p.weight.copy_(weights)
            model_p.bias.copy_(biases)
    
        y_pred = model(xs[0])[0]
        y_pred_p = model_p(torch.tensor(xs[0]))
    
>       assert math.isClose(y_pred.data, y_pred_p.item(), rel_tol=1e-5)
               ^^^^^^^^^^^^
E       AttributeError: module 'math' has no attribute 'isClose'. Did you mean: 'isclose'?

tests/gradient_test.py:85: AttributeError
=================================================================== short test summary info ===================================================================
FAILED tests/gradient_test.py::test_tanh - assert 0.010912398847672589 == -7.367088983301073e-05
FAILED tests/gradient_test.py::test_MLP_forward - AttributeError: module 'math' has no attribute 'isClose'. Did you mean: 'isclose'?
================================================================ 2 failed, 42 passed in 0.88s =================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py ...F.                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

========================================================================== FAILURES ===========================================================================
__________________________________________________________________________ test_tanh __________________________________________________________________________

    def test_tanh():
        a = Value(5.2)
        b = Value(0.5)
        c = (a * b).tanh()
        c.backward()
    
        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(-3.0, requires_grad=True)
        c_t = (a_t * b_t).tanh()
        c_t.backward()
    
>       assert a.grad == a_t.grad.item()
E       assert 0.010912398847672589 == -7.367088983301073e-05
E        +  where 0.010912398847672589 = Value(data=5.2).grad
E        +  and   -7.367088983301073e-05 = <built-in method item of Tensor object at 0x10dd259f0>()
E        +    where <built-in method item of Tensor object at 0x10dd259f0> = tensor(-7.3671e-05).item
E        +      where tensor(-7.3671e-05) = tensor(2., requires_grad=True).grad

tests/gradient_test.py:64: AssertionError
=================================================================== short test summary info ===================================================================
FAILED tests/gradient_test.py::test_tanh - assert 0.010912398847672589 == -7.367088983301073e-05
================================================================ 1 failed, 43 passed in 0.68s =================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py ...F.                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

========================================================================== FAILURES ===========================================================================
__________________________________________________________________________ test_tanh __________________________________________________________________________

    def test_tanh():
        a = Value(5.2)
        b = Value(0.5)
        c = (a * b).tanh()
        c.backward()
    
        a_t = torch.tensor(2.0, requires_grad=True)
        b_t = torch.tensor(-3.0, requires_grad=True)
        c_t = (a_t * b_t).tanh()
        c_t.backward()
    
>       assert a.grad == a_t.grad.item()
E       assert 0.010912398847672589 == -7.367088983301073e-05
E        +  where 0.010912398847672589 = Value(data=5.2).grad
E        +  and   -7.367088983301073e-05 = <built-in method item of Tensor object at 0x10df919a0>()
E        +    where <built-in method item of Tensor object at 0x10df919a0> = tensor(-7.3671e-05).item
E        +      where tensor(-7.3671e-05) = tensor(2., requires_grad=True).grad

tests/gradient_test.py:64: AssertionError
=================================================================== short test summary info ===================================================================
FAILED tests/gradient_test.py::test_tanh - assert 0.010912398847672589 == -7.367088983301073e-05
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py ...F.                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

========================================================================== FAILURES ===========================================================================
__________________________________________________________________________ test_tanh __________________________________________________________________________

    def test_tanh():
        a = Value(5.2)
        b = Value(0.5)
        c = (a * b).tanh()
        c.backward()
    
        a_t = torch.tensor(5.2, requires_grad=True)
        b_t = torch.tensor(0.5, requires_grad=True)
        c_t = (a_t * b_t).tanh()
        c_t.backward()
    
>       assert a.grad == a_t.grad.item()
E       assert 0.010912398847672589 == 0.010912420228123665
E        +  where 0.010912398847672589 = Value(data=5.2).grad
E        +  and   0.010912420228123665 = <built-in method item of Tensor object at 0x10d3b19f0>()
E        +    where <built-in method item of Tensor object at 0x10d3b19f0> = tensor(0.0109).item
E        +      where tensor(0.0109) = tensor(5.2000, requires_grad=True).grad

tests/gradient_test.py:64: AssertionError
=================================================================== short test summary info ===================================================================
FAILED tests/gradient_test.py::test_tanh - assert 0.010912398847672589 == 0.010912420228123665
================================================================ 1 failed, 43 passed in 0.87s =================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % kkkk[A
zsh: bad pattern: kkkk[A
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.83s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.86s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.90s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.92s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.86s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest 
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.80s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   src/mygrad/engine.py
        modified:   src/mygrad/nn/layers.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        scratch/note.py
        scratch/train_loop.py
        tests/gradient_test.py

no changes added to commit (use "git add" and/or "git commit -a")
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % git add .
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % git status
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   scratch/note.py
        new file:   scratch/train_loop.py
        modified:   src/mygrad/engine.py
        modified:   src/mygrad/nn/layers.py
        new file:   tests/gradient_test.py

(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % git commit -m "implement gradient test cases"
[main 3bbbe67] implement gradient test cases
 5 files changed, 135 insertions(+), 1 deletion(-)
 create mode 100644 scratch/note.py
 create mode 100644 scratch/train_loop.py
 create mode 100644 tests/gradient_test.py
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % git push
Enumerating objects: 20, done.
Counting objects: 100% (20/20), done.
Delta compression using up to 10 threads
Compressing objects: 100% (11/11), done.
Writing objects: 100% (12/12), 1.93 KiB | 1.93 MiB/s, done.
Total 12 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (3/3), completed with 3 local objects.
To https://github.com/kyleuphus/ml-library.git
   634a080..3bbbe67  main -> main
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 39 items / 1 error                                                                                                                                  

=========================================================================== ERRORS ============================================================================
___________________________________________________________ ERROR collecting tests/gradient_test.py ___________________________________________________________
ImportError while importing test module '/Users/kyleuphus/docs/ml-library/tests/gradient_test.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/gradient_test.py:4: in <module>
    from mygrad.nn import MLP
src/mygrad/nn/__init__.py:1: in <module>
    from .layers import Neuron, Layer, MLP
src/mygrad/nn/layers.py:3: in <module>
    from mygrad.nn.module import Module
E   ImportError: cannot import name 'Module' from 'mygrad.nn.module' (/Users/kyleuphus/docs/ml-library/src/mygrad/nn/module.py)
=================================================================== short test summary info ===================================================================
ERROR tests/gradient_test.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
====================================================================== 1 error in 0.84s =======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library % pytest
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/kyleuphus/docs/ml-library
configfile: pyproject.toml
collected 44 items                                                                                                                                            

tests/gradient_test.py .....                                                                                                                            [ 11%]
tests/test_engine.py .......................................                                                                                            [100%]

===================================================================== 44 passed in 0.89s ======================================================================
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library %  python3 scratch/train_loop.py
Epoch: 0, Loss: 8.042654940689173
Epoch: 1, Loss: 15.989987235902401
Epoch: 2, Loss: 24.182314355905568
Epoch: 3, Loss: 17.84620366251193
Epoch: 4, Loss: 0.5961298491663205
Epoch: 5, Loss: 0.27919155918399446
Epoch: 6, Loss: 0.21601376395514313
Epoch: 7, Loss: 0.19306340917005313
Epoch: 8, Loss: 0.18469284940229933
Epoch: 9, Loss: 0.18167512157582466
Epoch: 10, Loss: 0.18059559839211992
Epoch: 11, Loss: 0.18021127499569883
Epoch: 12, Loss: 0.1800748445395759
Epoch: 13, Loss: 0.18002649423981004
Epoch: 14, Loss: 0.18000937516566226
Epoch: 15, Loss: 0.1800033169208953
Epoch: 16, Loss: 0.18000117347446423
Epoch: 17, Loss: 0.18000041516904555
Epoch: 18, Loss: 0.18000014689454663
Epoch: 19, Loss: 0.1800000519785529
Epoch: 20, Loss: 0.1800000183943374
Epoch: 21, Loss: 0.18000000651008147
Epoch: 22, Loss: 0.18000000230425384
Epoch: 23, Loss: 0.18000000081566975
Epoch: 24, Loss: 0.18000000028875962
Epoch: 25, Loss: 0.18000000010223385
Epoch: 26, Loss: 0.1800000000361982
Epoch: 27, Loss: 0.18000000001281774
Epoch: 28, Loss: 0.18000000000453897
Epoch: 29, Loss: 0.1800000000016074
Epoch: 30, Loss: 0.18000000000056926
Epoch: 31, Loss: 0.18000000000020167
Epoch: 32, Loss: 0.1800000000000714
Epoch: 33, Loss: 0.18000000000002508
Epoch: 34, Loss: 0.180000000000009
Epoch: 35, Loss: 0.1800000000000032
Epoch: 36, Loss: 0.18000000000000116
Epoch: 37, Loss: 0.1800000000000005
Epoch: 38, Loss: 0.18000000000000008
Epoch: 39, Loss: 0.17999999999999977
Epoch: 40, Loss: 0.17999999999999988
Epoch: 41, Loss: 0.1800000000000002
Epoch: 42, Loss: 0.18000000000000024
Epoch: 43, Loss: 0.18000000000000016
Epoch: 44, Loss: 0.1800000000000002
Epoch: 45, Loss: 0.1799999999999998
Epoch: 46, Loss: 0.18000000000000002
Epoch: 47, Loss: 0.18000000000000016
Epoch: 48, Loss: 0.1800000000000002
Epoch: 49, Loss: 0.18000000000000022
Epoch: 50, Loss: 0.1799999999999998
Epoch: 51, Loss: 0.17999999999999983
Epoch: 52, Loss: 0.1799999999999998
Epoch: 53, Loss: 0.17999999999999994
Epoch: 54, Loss: 0.17999999999999988
Epoch: 55, Loss: 0.18000000000000005
Epoch: 56, Loss: 0.18000000000000005
Epoch: 57, Loss: 0.17999999999999994
Epoch: 58, Loss: 0.17999999999999994
Epoch: 59, Loss: 0.17999999999999997
Epoch: 60, Loss: 0.17999999999999997
Epoch: 61, Loss: 0.1799999999999999
Epoch: 62, Loss: 0.1800000000000001
Epoch: 63, Loss: 0.18000000000000016
Epoch: 64, Loss: 0.17999999999999988
Epoch: 65, Loss: 0.17999999999999997
Epoch: 66, Loss: 0.17999999999999994
Epoch: 67, Loss: 0.17999999999999985
Epoch: 68, Loss: 0.17999999999999977
Epoch: 69, Loss: 0.18
Epoch: 70, Loss: 0.17999999999999997
Epoch: 71, Loss: 0.1799999999999999
Epoch: 72, Loss: 0.18000000000000008
Epoch: 73, Loss: 0.18000000000000005
Epoch: 74, Loss: 0.1800000000000001
Epoch: 75, Loss: 0.18
Epoch: 76, Loss: 0.17999999999999994
Epoch: 77, Loss: 0.17999999999999994
Epoch: 78, Loss: 0.17999999999999994
Epoch: 79, Loss: 0.17999999999999994
Epoch: 80, Loss: 0.17999999999999994
Epoch: 81, Loss: 0.17999999999999994
Epoch: 82, Loss: 0.17999999999999994
Epoch: 83, Loss: 0.17999999999999994
Epoch: 84, Loss: 0.17999999999999994
Epoch: 85, Loss: 0.17999999999999994
Epoch: 86, Loss: 0.17999999999999994
Epoch: 87, Loss: 0.17999999999999994
Epoch: 88, Loss: 0.17999999999999994
Epoch: 89, Loss: 0.17999999999999994
Epoch: 90, Loss: 0.1799999999999999
Epoch: 91, Loss: 0.1799999999999999
Epoch: 92, Loss: 0.1799999999999999
Epoch: 93, Loss: 0.18
Epoch: 94, Loss: 0.18
Epoch: 95, Loss: 0.17999999999999997
Epoch: 96, Loss: 0.18
Epoch: 97, Loss: 0.17999999999999997
Epoch: 98, Loss: 0.18
Epoch: 99, Loss: 0.17999999999999997
Epoch: 100, Loss: 0.18000000000000005
Epoch: 101, Loss: 0.18000000000000008
Epoch: 102, Loss: 0.18000000000000005
Epoch: 103, Loss: 0.18000000000000008
Epoch: 104, Loss: 0.18000000000000005
Epoch: 105, Loss: 0.18000000000000008
Epoch: 106, Loss: 0.18000000000000005
Epoch: 107, Loss: 0.18000000000000008
Epoch: 108, Loss: 0.18000000000000005
Epoch: 109, Loss: 0.18
Epoch: 110, Loss: 0.17999999999999997
Epoch: 111, Loss: 0.17999999999999997
Epoch: 112, Loss: 0.18
Epoch: 113, Loss: 0.17999999999999997
Epoch: 114, Loss: 0.17999999999999997
Epoch: 115, Loss: 0.18
Epoch: 116, Loss: 0.17999999999999997
Epoch: 117, Loss: 0.17999999999999997
Epoch: 118, Loss: 0.18
Epoch: 119, Loss: 0.17999999999999997
Epoch: 120, Loss: 0.17999999999999997
Epoch: 121, Loss: 0.18
Epoch: 122, Loss: 0.17999999999999997
Epoch: 123, Loss: 0.17999999999999997
Epoch: 124, Loss: 0.18
Epoch: 125, Loss: 0.17999999999999997
Epoch: 126, Loss: 0.17999999999999997
Epoch: 127, Loss: 0.18
Epoch: 128, Loss: 0.17999999999999997
Epoch: 129, Loss: 0.17999999999999997
Epoch: 130, Loss: 0.18
Epoch: 131, Loss: 0.17999999999999988
Epoch: 132, Loss: 0.17999999999999988
Epoch: 133, Loss: 0.17999999999999988
Epoch: 134, Loss: 0.17999999999999988
Epoch: 135, Loss: 0.17999999999999988
Epoch: 136, Loss: 0.17999999999999988
Epoch: 137, Loss: 0.17999999999999988
Epoch: 138, Loss: 0.17999999999999988
Epoch: 139, Loss: 0.17999999999999988
Epoch: 140, Loss: 0.17999999999999988
Epoch: 141, Loss: 0.17999999999999988
Epoch: 142, Loss: 0.17999999999999988
Epoch: 143, Loss: 0.17999999999999988
Epoch: 144, Loss: 0.17999999999999988
Epoch: 145, Loss: 0.17999999999999988
Epoch: 146, Loss: 0.17999999999999994
Epoch: 147, Loss: 0.17999999999999994
Epoch: 148, Loss: 0.17999999999999994
Epoch: 149, Loss: 0.17999999999999994
Epoch: 150, Loss: 0.1799999999999999
Epoch: 151, Loss: 0.1799999999999999
Epoch: 152, Loss: 0.1799999999999999
Epoch: 153, Loss: 0.1799999999999999
Epoch: 154, Loss: 0.1799999999999999
Epoch: 155, Loss: 0.17999999999999988
Epoch: 156, Loss: 0.17999999999999988
Epoch: 157, Loss: 0.17999999999999988
Epoch: 158, Loss: 0.17999999999999988
Epoch: 159, Loss: 0.17999999999999988
Epoch: 160, Loss: 0.18000000000000005
Epoch: 161, Loss: 0.17999999999999997
Epoch: 162, Loss: 0.17999999999999997
Epoch: 163, Loss: 0.17999999999999997
Epoch: 164, Loss: 0.17999999999999997
Epoch: 165, Loss: 0.17999999999999997
Epoch: 166, Loss: 0.17999999999999997
Epoch: 167, Loss: 0.17999999999999997
Epoch: 168, Loss: 0.17999999999999997
Epoch: 169, Loss: 0.17999999999999997
Epoch: 170, Loss: 0.17999999999999997
Epoch: 171, Loss: 0.17999999999999997
Epoch: 172, Loss: 0.17999999999999997
Epoch: 173, Loss: 0.17999999999999997
Epoch: 174, Loss: 0.17999999999999997
Epoch: 175, Loss: 0.17999999999999997
Epoch: 176, Loss: 0.17999999999999997
Epoch: 177, Loss: 0.17999999999999997
Epoch: 178, Loss: 0.17999999999999997
Epoch: 179, Loss: 0.17999999999999997
Epoch: 180, Loss: 0.17999999999999997
Epoch: 181, Loss: 0.17999999999999997
Epoch: 182, Loss: 0.17999999999999997
Epoch: 183, Loss: 0.17999999999999997
Epoch: 184, Loss: 0.17999999999999997
Epoch: 185, Loss: 0.17999999999999997
Epoch: 186, Loss: 0.17999999999999997
Epoch: 187, Loss: 0.17999999999999997
Epoch: 188, Loss: 0.17999999999999997
Epoch: 189, Loss: 0.17999999999999997
Epoch: 190, Loss: 0.17999999999999997
Epoch: 191, Loss: 0.17999999999999997
Epoch: 192, Loss: 0.17999999999999997
Epoch: 193, Loss: 0.17999999999999997
Epoch: 194, Loss: 0.17999999999999997
Epoch: 195, Loss: 0.17999999999999997
Epoch: 196, Loss: 0.17999999999999997
Epoch: 197, Loss: 0.17999999999999997
Epoch: 198, Loss: 0.17999999999999997
Epoch: 199, Loss: 0.17999999999999997
Epoch: 200, Loss: 0.17999999999999997
Epoch: 201, Loss: 0.17999999999999997
Epoch: 202, Loss: 0.17999999999999997
Epoch: 203, Loss: 0.17999999999999997
Epoch: 204, Loss: 0.17999999999999997
Epoch: 205, Loss: 0.17999999999999997
Epoch: 206, Loss: 0.17999999999999997
Epoch: 207, Loss: 0.17999999999999997
Epoch: 208, Loss: 0.17999999999999997
Epoch: 209, Loss: 0.17999999999999997
Epoch: 210, Loss: 0.17999999999999997
Epoch: 211, Loss: 0.17999999999999997
Epoch: 212, Loss: 0.17999999999999997
Epoch: 213, Loss: 0.17999999999999997
Epoch: 214, Loss: 0.17999999999999997
Epoch: 215, Loss: 0.17999999999999997
Epoch: 216, Loss: 0.17999999999999997
Epoch: 217, Loss: 0.17999999999999997
Epoch: 218, Loss: 0.17999999999999997
Epoch: 219, Loss: 0.17999999999999997
Epoch: 220, Loss: 0.17999999999999997
Epoch: 221, Loss: 0.17999999999999997
Epoch: 222, Loss: 0.17999999999999997
Epoch: 223, Loss: 0.17999999999999997
Epoch: 224, Loss: 0.17999999999999997
Epoch: 225, Loss: 0.17999999999999997
Epoch: 226, Loss: 0.17999999999999997
Epoch: 227, Loss: 0.17999999999999997
Epoch: 228, Loss: 0.17999999999999997
Epoch: 229, Loss: 0.17999999999999997
Epoch: 230, Loss: 0.17999999999999997
Epoch: 231, Loss: 0.17999999999999997
Epoch: 232, Loss: 0.17999999999999997
Epoch: 233, Loss: 0.17999999999999997
Epoch: 234, Loss: 0.17999999999999997
Epoch: 235, Loss: 0.17999999999999997
Epoch: 236, Loss: 0.17999999999999997
Epoch: 237, Loss: 0.17999999999999997
Epoch: 238, Loss: 0.17999999999999997
Epoch: 239, Loss: 0.17999999999999997
Epoch: 240, Loss: 0.17999999999999997
Epoch: 241, Loss: 0.17999999999999997
Epoch: 242, Loss: 0.17999999999999997
Epoch: 243, Loss: 0.17999999999999997
Epoch: 244, Loss: 0.17999999999999997
Epoch: 245, Loss: 0.18
Epoch: 246, Loss: 0.17999999999999997
Epoch: 247, Loss: 0.17999999999999997
Epoch: 248, Loss: 0.17999999999999997
Epoch: 249, Loss: 0.18
Epoch: 250, Loss: 0.17999999999999997
Epoch: 251, Loss: 0.17999999999999997
Epoch: 252, Loss: 0.17999999999999997
Epoch: 253, Loss: 0.17999999999999997
Epoch: 254, Loss: 0.17999999999999997
Epoch: 255, Loss: 0.17999999999999997
Epoch: 256, Loss: 0.17999999999999997
Epoch: 257, Loss: 0.17999999999999997
Epoch: 258, Loss: 0.17999999999999997
Epoch: 259, Loss: 0.17999999999999997
Epoch: 260, Loss: 0.17999999999999997
Epoch: 261, Loss: 0.17999999999999997
Epoch: 262, Loss: 0.17999999999999997
Epoch: 263, Loss: 0.17999999999999997
Epoch: 264, Loss: 0.17999999999999997
Epoch: 265, Loss: 0.17999999999999997
Epoch: 266, Loss: 0.17999999999999997
Epoch: 267, Loss: 0.17999999999999997
Epoch: 268, Loss: 0.17999999999999997
Epoch: 269, Loss: 0.17999999999999994
Epoch: 270, Loss: 0.17999999999999997
Epoch: 271, Loss: 0.17999999999999997
Epoch: 272, Loss: 0.17999999999999997
Epoch: 273, Loss: 0.17999999999999997
Epoch: 274, Loss: 0.17999999999999997
Epoch: 275, Loss: 0.17999999999999997
Epoch: 276, Loss: 0.17999999999999997
Epoch: 277, Loss: 0.17999999999999997
Epoch: 278, Loss: 0.18000000000000005
Epoch: 279, Loss: 0.17999999999999994
Epoch: 280, Loss: 0.17999999999999994
Epoch: 281, Loss: 0.17999999999999994
Epoch: 282, Loss: 0.17999999999999994
Epoch: 283, Loss: 0.17999999999999994
Epoch: 284, Loss: 0.17999999999999994
Epoch: 285, Loss: 0.17999999999999994
Epoch: 286, Loss: 0.17999999999999994
Epoch: 287, Loss: 0.17999999999999994
Epoch: 288, Loss: 0.17999999999999994
Epoch: 289, Loss: 0.17999999999999994
Epoch: 290, Loss: 0.17999999999999994
Epoch: 291, Loss: 0.17999999999999994
Epoch: 292, Loss: 0.18
Epoch: 293, Loss: 0.17999999999999994
Epoch: 294, Loss: 0.18
Epoch: 295, Loss: 0.18
Epoch: 296, Loss: 0.17999999999999994
Epoch: 297, Loss: 0.18
Epoch: 298, Loss: 0.18
Epoch: 299, Loss: 0.17999999999999994
Epoch: 300, Loss: 0.18
Epoch: 301, Loss: 0.18
Epoch: 302, Loss: 0.17999999999999994
Epoch: 303, Loss: 0.18
Epoch: 304, Loss: 0.17999999999999994
Epoch: 305, Loss: 0.18
Epoch: 306, Loss: 0.18
Epoch: 307, Loss: 0.17999999999999994
Epoch: 308, Loss: 0.18
Epoch: 309, Loss: 0.18
Epoch: 310, Loss: 0.17999999999999994
Epoch: 311, Loss: 0.18
Epoch: 312, Loss: 0.17999999999999994
Epoch: 313, Loss: 0.18
Epoch: 314, Loss: 0.17999999999999994
Epoch: 315, Loss: 0.18
Epoch: 316, Loss: 0.18
Epoch: 317, Loss: 0.17999999999999994
Epoch: 318, Loss: 0.18
Epoch: 319, Loss: 0.17999999999999994
Epoch: 320, Loss: 0.18
Epoch: 321, Loss: 0.17999999999999994
Epoch: 322, Loss: 0.18
Epoch: 323, Loss: 0.17999999999999994
Epoch: 324, Loss: 0.18
Epoch: 325, Loss: 0.18
Epoch: 326, Loss: 0.17999999999999994
Epoch: 327, Loss: 0.18
Epoch: 328, Loss: 0.17999999999999994
Epoch: 329, Loss: 0.18
Epoch: 330, Loss: 0.18
Epoch: 331, Loss: 0.17999999999999994
Epoch: 332, Loss: 0.18
Epoch: 333, Loss: 0.18
Epoch: 334, Loss: 0.17999999999999994
Epoch: 335, Loss: 0.18
Epoch: 336, Loss: 0.17999999999999994
Epoch: 337, Loss: 0.18
Epoch: 338, Loss: 0.17999999999999994
Epoch: 339, Loss: 0.18
Epoch: 340, Loss: 0.17999999999999994
Epoch: 341, Loss: 0.18
Epoch: 342, Loss: 0.17999999999999994
Epoch: 343, Loss: 0.18
Epoch: 344, Loss: 0.18
Epoch: 345, Loss: 0.17999999999999994
Epoch: 346, Loss: 0.18
Epoch: 347, Loss: 0.17999999999999994
Epoch: 348, Loss: 0.18
Epoch: 349, Loss: 0.18
Epoch: 350, Loss: 0.17999999999999994
Epoch: 351, Loss: 0.18
Epoch: 352, Loss: 0.18
Epoch: 353, Loss: 0.17999999999999994
Epoch: 354, Loss: 0.18
Epoch: 355, Loss: 0.17999999999999994
Epoch: 356, Loss: 0.18
Epoch: 357, Loss: 0.18
Epoch: 358, Loss: 0.17999999999999994
Epoch: 359, Loss: 0.18
Epoch: 360, Loss: 0.17999999999999994
Epoch: 361, Loss: 0.18
Epoch: 362, Loss: 0.18
Epoch: 363, Loss: 0.17999999999999994
Epoch: 364, Loss: 0.18
Epoch: 365, Loss: 0.17999999999999994
Epoch: 366, Loss: 0.18
Epoch: 367, Loss: 0.17999999999999994
Epoch: 368, Loss: 0.18
Epoch: 369, Loss: 0.18
Epoch: 370, Loss: 0.17999999999999994
Epoch: 371, Loss: 0.18
Epoch: 372, Loss: 0.17999999999999994
Epoch: 373, Loss: 0.18
Epoch: 374, Loss: 0.17999999999999994
Epoch: 375, Loss: 0.18
Epoch: 376, Loss: 0.17999999999999994
Epoch: 377, Loss: 0.18
Epoch: 378, Loss: 0.18
Epoch: 379, Loss: 0.17999999999999994
Epoch: 380, Loss: 0.18
Epoch: 381, Loss: 0.17999999999999994
Epoch: 382, Loss: 0.18
Epoch: 383, Loss: 0.17999999999999994
Epoch: 384, Loss: 0.18
Epoch: 385, Loss: 0.17999999999999994
Epoch: 386, Loss: 0.18
Epoch: 387, Loss: 0.18
Epoch: 388, Loss: 0.17999999999999994
Epoch: 389, Loss: 0.18
Epoch: 390, Loss: 0.17999999999999994
Epoch: 391, Loss: 0.18
Epoch: 392, Loss: 0.17999999999999994
Epoch: 393, Loss: 0.18
Epoch: 394, Loss: 0.17999999999999994
Epoch: 395, Loss: 0.18
Epoch: 396, Loss: 0.18
Epoch: 397, Loss: 0.17999999999999994
Epoch: 398, Loss: 0.18
Epoch: 399, Loss: 0.17999999999999994
Epoch: 400, Loss: 0.18
Epoch: 401, Loss: 0.18
Epoch: 402, Loss: 0.17999999999999994
Epoch: 403, Loss: 0.18
Epoch: 404, Loss: 0.18
Epoch: 405, Loss: 0.17999999999999994
Epoch: 406, Loss: 0.18
Epoch: 407, Loss: 0.17999999999999994
Epoch: 408, Loss: 0.18
Epoch: 409, Loss: 0.18
Epoch: 410, Loss: 0.17999999999999994
Epoch: 411, Loss: 0.18
Epoch: 412, Loss: 0.18
Epoch: 413, Loss: 0.17999999999999994
Epoch: 414, Loss: 0.18
Epoch: 415, Loss: 0.18
Epoch: 416, Loss: 0.17999999999999994
Epoch: 417, Loss: 0.18
Epoch: 418, Loss: 0.17999999999999994
Epoch: 419, Loss: 0.18
Epoch: 420, Loss: 0.17999999999999994
Epoch: 421, Loss: 0.18
Epoch: 422, Loss: 0.18
Epoch: 423, Loss: 0.17999999999999994
Epoch: 424, Loss: 0.18
Epoch: 425, Loss: 0.17999999999999994
Epoch: 426, Loss: 0.18
Epoch: 427, Loss: 0.17999999999999994
Epoch: 428, Loss: 0.18
Epoch: 429, Loss: 0.17999999999999994
Epoch: 430, Loss: 0.18
Epoch: 431, Loss: 0.18
Epoch: 432, Loss: 0.17999999999999994
Epoch: 433, Loss: 0.18
Epoch: 434, Loss: 0.17999999999999994
Epoch: 435, Loss: 0.18
Epoch: 436, Loss: 0.17999999999999994
Epoch: 437, Loss: 0.18
Epoch: 438, Loss: 0.17999999999999994
Epoch: 439, Loss: 0.18
Epoch: 440, Loss: 0.18
Epoch: 441, Loss: 0.17999999999999994
Epoch: 442, Loss: 0.18
Epoch: 443, Loss: 0.17999999999999994
Epoch: 444, Loss: 0.18
Epoch: 445, Loss: 0.17999999999999994
Epoch: 446, Loss: 0.18
Epoch: 447, Loss: 0.17999999999999994
Epoch: 448, Loss: 0.18
Epoch: 449, Loss: 0.18
Epoch: 450, Loss: 0.17999999999999994
Epoch: 451, Loss: 0.18
Epoch: 452, Loss: 0.17999999999999994
Epoch: 453, Loss: 0.18
Epoch: 454, Loss: 0.18
Epoch: 455, Loss: 0.17999999999999994
Epoch: 456, Loss: 0.18
Epoch: 457, Loss: 0.18
Epoch: 458, Loss: 0.17999999999999994
Epoch: 459, Loss: 0.18
Epoch: 460, Loss: 0.18
Epoch: 461, Loss: 0.17999999999999994
Epoch: 462, Loss: 0.18
Epoch: 463, Loss: 0.17999999999999994
Epoch: 464, Loss: 0.18
Epoch: 465, Loss: 0.18
Epoch: 466, Loss: 0.17999999999999994
Epoch: 467, Loss: 0.18
Epoch: 468, Loss: 0.18
Epoch: 469, Loss: 0.17999999999999994
Epoch: 470, Loss: 0.18
Epoch: 471, Loss: 0.17999999999999994
Epoch: 472, Loss: 0.18
Epoch: 473, Loss: 0.17999999999999994
Epoch: 474, Loss: 0.18
Epoch: 475, Loss: 0.18
Epoch: 476, Loss: 0.17999999999999994
Epoch: 477, Loss: 0.18
Epoch: 478, Loss: 0.17999999999999994
Epoch: 479, Loss: 0.18
Epoch: 480, Loss: 0.17999999999999994
Epoch: 481, Loss: 0.18
Epoch: 482, Loss: 0.17999999999999994
Epoch: 483, Loss: 0.18
Epoch: 484, Loss: 0.18
Epoch: 485, Loss: 0.17999999999999994
Epoch: 486, Loss: 0.18
Epoch: 487, Loss: 0.17999999999999994
Epoch: 488, Loss: 0.18
Epoch: 489, Loss: 0.17999999999999994
Epoch: 490, Loss: 0.18
Epoch: 491, Loss: 0.17999999999999994
Epoch: 492, Loss: 0.18
Epoch: 493, Loss: 0.18
Epoch: 494, Loss: 0.17999999999999994
Epoch: 495, Loss: 0.18
Epoch: 496, Loss: 0.17999999999999994
Epoch: 497, Loss: 0.18
Epoch: 498, Loss: 0.17999999999999994
Epoch: 499, Loss: 0.18
Epoch: 500, Loss: 0.17999999999999994
Epoch: 501, Loss: 0.18
Epoch: 502, Loss: 0.18
Epoch: 503, Loss: 0.17999999999999994
Epoch: 504, Loss: 0.18
Epoch: 505, Loss: 0.17999999999999994
Epoch: 506, Loss: 0.18
Epoch: 507, Loss: 0.18
Epoch: 508, Loss: 0.17999999999999994
Epoch: 509, Loss: 0.18
Epoch: 510, Loss: 0.18
Epoch: 511, Loss: 0.17999999999999994
Epoch: 512, Loss: 0.18
Epoch: 513, Loss: 0.17999999999999994
Epoch: 514, Loss: 0.18
Epoch: 515, Loss: 0.18
Epoch: 516, Loss: 0.17999999999999994
Epoch: 517, Loss: 0.18
Epoch: 518, Loss: 0.18
Epoch: 519, Loss: 0.17999999999999994
Epoch: 520, Loss: 0.18
Epoch: 521, Loss: 0.18
Epoch: 522, Loss: 0.17999999999999994
Epoch: 523, Loss: 0.18
Epoch: 524, Loss: 0.17999999999999994
Epoch: 525, Loss: 0.18
Epoch: 526, Loss: 0.17999999999999994
Epoch: 527, Loss: 0.18
Epoch: 528, Loss: 0.18
Epoch: 529, Loss: 0.17999999999999994
Epoch: 530, Loss: 0.18
Epoch: 531, Loss: 0.17999999999999994
Epoch: 532, Loss: 0.18
Epoch: 533, Loss: 0.17999999999999994
Epoch: 534, Loss: 0.18
Epoch: 535, Loss: 0.17999999999999994
Epoch: 536, Loss: 0.18
Epoch: 537, Loss: 0.18
Epoch: 538, Loss: 0.17999999999999994
Epoch: 539, Loss: 0.18
Epoch: 540, Loss: 0.17999999999999994
Epoch: 541, Loss: 0.18
Epoch: 542, Loss: 0.17999999999999994
Epoch: 543, Loss: 0.18
Epoch: 544, Loss: 0.17999999999999994
Epoch: 545, Loss: 0.18
Epoch: 546, Loss: 0.18
Epoch: 547, Loss: 0.17999999999999994
Epoch: 548, Loss: 0.18
Epoch: 549, Loss: 0.17999999999999994
Epoch: 550, Loss: 0.18
Epoch: 551, Loss: 0.17999999999999994
Epoch: 552, Loss: 0.18
Epoch: 553, Loss: 0.17999999999999994
Epoch: 554, Loss: 0.18
Epoch: 555, Loss: 0.18
Epoch: 556, Loss: 0.17999999999999994
Epoch: 557, Loss: 0.18
Epoch: 558, Loss: 0.17999999999999994
Epoch: 559, Loss: 0.18
Epoch: 560, Loss: 0.18
Epoch: 561, Loss: 0.17999999999999994
Epoch: 562, Loss: 0.18
Epoch: 563, Loss: 0.18
Epoch: 564, Loss: 0.17999999999999994
Epoch: 565, Loss: 0.18
Epoch: 566, Loss: 0.18
Epoch: 567, Loss: 0.17999999999999994
Epoch: 568, Loss: 0.18
Epoch: 569, Loss: 0.17999999999999994
Epoch: 570, Loss: 0.18
Epoch: 571, Loss: 0.18
Epoch: 572, Loss: 0.17999999999999994
Epoch: 573, Loss: 0.18
Epoch: 574, Loss: 0.18
Epoch: 575, Loss: 0.17999999999999994
Epoch: 576, Loss: 0.18
Epoch: 577, Loss: 0.17999999999999994
Epoch: 578, Loss: 0.18
Epoch: 579, Loss: 0.17999999999999994
Epoch: 580, Loss: 0.18
Epoch: 581, Loss: 0.18
Epoch: 582, Loss: 0.17999999999999994
Epoch: 583, Loss: 0.18
Epoch: 584, Loss: 0.17999999999999994
Epoch: 585, Loss: 0.18
Epoch: 586, Loss: 0.17999999999999994
Epoch: 587, Loss: 0.18
Epoch: 588, Loss: 0.17999999999999994
Epoch: 589, Loss: 0.18
Epoch: 590, Loss: 0.18
Epoch: 591, Loss: 0.17999999999999994
Epoch: 592, Loss: 0.18
Epoch: 593, Loss: 0.17999999999999994
Epoch: 594, Loss: 0.18
Epoch: 595, Loss: 0.17999999999999994
Epoch: 596, Loss: 0.18
Epoch: 597, Loss: 0.17999999999999994
Epoch: 598, Loss: 0.18
Epoch: 599, Loss: 0.18
Epoch: 600, Loss: 0.17999999999999994
Epoch: 601, Loss: 0.18
Epoch: 602, Loss: 0.17999999999999994
Epoch: 603, Loss: 0.18
Epoch: 604, Loss: 0.17999999999999994
Epoch: 605, Loss: 0.18
Epoch: 606, Loss: 0.17999999999999994
Epoch: 607, Loss: 0.18
Epoch: 608, Loss: 0.18
Epoch: 609, Loss: 0.17999999999999994
Epoch: 610, Loss: 0.18
Epoch: 611, Loss: 0.17999999999999994
Epoch: 612, Loss: 0.18
Epoch: 613, Loss: 0.18
Epoch: 614, Loss: 0.17999999999999994
Epoch: 615, Loss: 0.18
Epoch: 616, Loss: 0.18
Epoch: 617, Loss: 0.17999999999999994
Epoch: 618, Loss: 0.18
Epoch: 619, Loss: 0.18
Epoch: 620, Loss: 0.17999999999999994
Epoch: 621, Loss: 0.18
Epoch: 622, Loss: 0.17999999999999994
Epoch: 623, Loss: 0.18
Epoch: 624, Loss: 0.18
Epoch: 625, Loss: 0.17999999999999994
Epoch: 626, Loss: 0.18
Epoch: 627, Loss: 0.18
Epoch: 628, Loss: 0.17999999999999994
Epoch: 629, Loss: 0.18
Epoch: 630, Loss: 0.18
Epoch: 631, Loss: 0.17999999999999994
Epoch: 632, Loss: 0.18
Epoch: 633, Loss: 0.17999999999999994
Epoch: 634, Loss: 0.18
Epoch: 635, Loss: 0.18
Epoch: 636, Loss: 0.17999999999999994
Epoch: 637, Loss: 0.18
Epoch: 638, Loss: 0.17999999999999994
Epoch: 639, Loss: 0.18
Epoch: 640, Loss: 0.17999999999999994
Epoch: 641, Loss: 0.18
Epoch: 642, Loss: 0.18
Epoch: 643, Loss: 0.17999999999999994
Epoch: 644, Loss: 0.18
Epoch: 645, Loss: 0.17999999999999994
Epoch: 646, Loss: 0.18
Epoch: 647, Loss: 0.18
Epoch: 648, Loss: 0.17999999999999994
Epoch: 649, Loss: 0.18
Epoch: 650, Loss: 0.17999999999999994
Epoch: 651, Loss: 0.18
Epoch: 652, Loss: 0.18
Epoch: 653, Loss: 0.17999999999999994
Epoch: 654, Loss: 0.18
Epoch: 655, Loss: 0.17999999999999994
Epoch: 656, Loss: 0.18
Epoch: 657, Loss: 0.17999999999999994
Epoch: 658, Loss: 0.18
Epoch: 659, Loss: 0.17999999999999994
Epoch: 660, Loss: 0.18
Epoch: 661, Loss: 0.18
Epoch: 662, Loss: 0.17999999999999994
Epoch: 663, Loss: 0.18
Epoch: 664, Loss: 0.17999999999999994
Epoch: 665, Loss: 0.18
Epoch: 666, Loss: 0.17999999999999994
Epoch: 667, Loss: 0.18
Epoch: 668, Loss: 0.18
Epoch: 669, Loss: 0.17999999999999994
Epoch: 670, Loss: 0.18
Epoch: 671, Loss: 0.18
Epoch: 672, Loss: 0.17999999999999994
Epoch: 673, Loss: 0.18
Epoch: 674, Loss: 0.18
Epoch: 675, Loss: 0.17999999999999994
Epoch: 676, Loss: 0.18
Epoch: 677, Loss: 0.18
Epoch: 678, Loss: 0.17999999999999994
Epoch: 679, Loss: 0.18
Epoch: 680, Loss: 0.18
Epoch: 681, Loss: 0.17999999999999994
Epoch: 682, Loss: 0.18
Epoch: 683, Loss: 0.17999999999999994
Epoch: 684, Loss: 0.18
Epoch: 685, Loss: 0.17999999999999994
Epoch: 686, Loss: 0.18
Epoch: 687, Loss: 0.18
Epoch: 688, Loss: 0.17999999999999994
Epoch: 689, Loss: 0.18
Epoch: 690, Loss: 0.17999999999999994
Epoch: 691, Loss: 0.18
Epoch: 692, Loss: 0.17999999999999994
Epoch: 693, Loss: 0.18
Epoch: 694, Loss: 0.17999999999999994
Epoch: 695, Loss: 0.18
Epoch: 696, Loss: 0.18
Epoch: 697, Loss: 0.17999999999999994
Epoch: 698, Loss: 0.18
Epoch: 699, Loss: 0.17999999999999994
Epoch: 700, Loss: 0.18
Epoch: 701, Loss: 0.17999999999999994
Epoch: 702, Loss: 0.18
Epoch: 703, Loss: 0.17999999999999994
Epoch: 704, Loss: 0.18
Epoch: 705, Loss: 0.17999999999999994
Epoch: 706, Loss: 0.18
Epoch: 707, Loss: 0.17999999999999994
Epoch: 708, Loss: 0.18
Epoch: 709, Loss: 0.17999999999999994
Epoch: 710, Loss: 0.18
Epoch: 711, Loss: 0.18
Epoch: 712, Loss: 0.17999999999999994
Epoch: 713, Loss: 0.18
Epoch: 714, Loss: 0.18
Epoch: 715, Loss: 0.17999999999999994
Epoch: 716, Loss: 0.18
Epoch: 717, Loss: 0.17999999999999994
Epoch: 718, Loss: 0.18
Epoch: 719, Loss: 0.18
Epoch: 720, Loss: 0.17999999999999994
Epoch: 721, Loss: 0.18
Epoch: 722, Loss: 0.18
Epoch: 723, Loss: 0.17999999999999994
Epoch: 724, Loss: 0.18
Epoch: 725, Loss: 0.18
Epoch: 726, Loss: 0.17999999999999994
Epoch: 727, Loss: 0.18
Epoch: 728, Loss: 0.17999999999999994
Epoch: 729, Loss: 0.18
Epoch: 730, Loss: 0.18
Epoch: 731, Loss: 0.17999999999999994
Epoch: 732, Loss: 0.18
Epoch: 733, Loss: 0.17999999999999994
Epoch: 734, Loss: 0.18
Epoch: 735, Loss: 0.17999999999999994
Epoch: 736, Loss: 0.18
Epoch: 737, Loss: 0.17999999999999994
Epoch: 738, Loss: 0.18
Epoch: 739, Loss: 0.17999999999999994
Epoch: 740, Loss: 0.18
Epoch: 741, Loss: 0.18
Epoch: 742, Loss: 0.17999999999999994
Epoch: 743, Loss: 0.18
Epoch: 744, Loss: 0.17999999999999994
Epoch: 745, Loss: 0.18
Epoch: 746, Loss: 0.17999999999999994
Epoch: 747, Loss: 0.18
Epoch: 748, Loss: 0.17999999999999994
Epoch: 749, Loss: 0.18
Epoch: 750, Loss: 0.17999999999999994
Epoch: 751, Loss: 0.18
Epoch: 752, Loss: 0.17999999999999994
Epoch: 753, Loss: 0.18
Epoch: 754, Loss: 0.17999999999999994
Epoch: 755, Loss: 0.18
Epoch: 756, Loss: 0.17999999999999994
Epoch: 757, Loss: 0.18
Epoch: 758, Loss: 0.17999999999999994
Epoch: 759, Loss: 0.18
Epoch: 760, Loss: 0.17999999999999994
Epoch: 761, Loss: 0.18
Epoch: 762, Loss: 0.17999999999999994
Epoch: 763, Loss: 0.18
Epoch: 764, Loss: 0.17999999999999994
Epoch: 765, Loss: 0.18
Epoch: 766, Loss: 0.17999999999999994
Epoch: 767, Loss: 0.18
Epoch: 768, Loss: 0.18
Epoch: 769, Loss: 0.17999999999999994
Epoch: 770, Loss: 0.18
Epoch: 771, Loss: 0.17999999999999994
Epoch: 772, Loss: 0.18
Epoch: 773, Loss: 0.18
Epoch: 774, Loss: 0.17999999999999994
Epoch: 775, Loss: 0.18
Epoch: 776, Loss: 0.18
Epoch: 777, Loss: 0.17999999999999994
Epoch: 778, Loss: 0.18
Epoch: 779, Loss: 0.17999999999999994
Epoch: 780, Loss: 0.18
Epoch: 781, Loss: 0.17999999999999994
Epoch: 782, Loss: 0.18
Epoch: 783, Loss: 0.18
Epoch: 784, Loss: 0.17999999999999994
Epoch: 785, Loss: 0.18
Epoch: 786, Loss: 0.17999999999999994
Epoch: 787, Loss: 0.18
Epoch: 788, Loss: 0.18
Epoch: 789, Loss: 0.17999999999999994
Epoch: 790, Loss: 0.18
Epoch: 791, Loss: 0.17999999999999994
Epoch: 792, Loss: 0.18
Epoch: 793, Loss: 0.17999999999999994
Epoch: 794, Loss: 0.18
Epoch: 795, Loss: 0.17999999999999994
Epoch: 796, Loss: 0.18
Epoch: 797, Loss: 0.17999999999999994
Epoch: 798, Loss: 0.18
Epoch: 799, Loss: 0.17999999999999994
Epoch: 800, Loss: 0.18
Epoch: 801, Loss: 0.18
Epoch: 802, Loss: 0.17999999999999994
Epoch: 803, Loss: 0.18
Epoch: 804, Loss: 0.17999999999999994
Epoch: 805, Loss: 0.18
Epoch: 806, Loss: 0.17999999999999994
Epoch: 807, Loss: 0.18
Epoch: 808, Loss: 0.18
Epoch: 809, Loss: 0.17999999999999994
Epoch: 810, Loss: 0.18
Epoch: 811, Loss: 0.18
Epoch: 812, Loss: 0.17999999999999994
Epoch: 813, Loss: 0.18
Epoch: 814, Loss: 0.17999999999999994
Epoch: 815, Loss: 0.18
Epoch: 816, Loss: 0.17999999999999994
Epoch: 817, Loss: 0.18
Epoch: 818, Loss: 0.17999999999999994
Epoch: 819, Loss: 0.18
Epoch: 820, Loss: 0.18
Epoch: 821, Loss: 0.17999999999999994
Epoch: 822, Loss: 0.18
Epoch: 823, Loss: 0.18
Epoch: 824, Loss: 0.17999999999999994
Epoch: 825, Loss: 0.18
Epoch: 826, Loss: 0.18
Epoch: 827, Loss: 0.17999999999999994
Epoch: 828, Loss: 0.18
Epoch: 829, Loss: 0.17999999999999994
Epoch: 830, Loss: 0.18
Epoch: 831, Loss: 0.18
Epoch: 832, Loss: 0.17999999999999994
Epoch: 833, Loss: 0.18
Epoch: 834, Loss: 0.18
Epoch: 835, Loss: 0.17999999999999994
Epoch: 836, Loss: 0.18
Epoch: 837, Loss: 0.17999999999999994
Epoch: 838, Loss: 0.18
Epoch: 839, Loss: 0.17999999999999994
Epoch: 840, Loss: 0.18
Epoch: 841, Loss: 0.17999999999999994
Epoch: 842, Loss: 0.18
Epoch: 843, Loss: 0.18
Epoch: 844, Loss: 0.17999999999999994
Epoch: 845, Loss: 0.18
Epoch: 846, Loss: 0.17999999999999994
Epoch: 847, Loss: 0.18
Epoch: 848, Loss: 0.18
Epoch: 849, Loss: 0.17999999999999994
Epoch: 850, Loss: 0.18
Epoch: 851, Loss: 0.17999999999999994
Epoch: 852, Loss: 0.18
Epoch: 853, Loss: 0.17999999999999994
Epoch: 854, Loss: 0.18
Epoch: 855, Loss: 0.18
Epoch: 856, Loss: 0.17999999999999994
Epoch: 857, Loss: 0.18
Epoch: 858, Loss: 0.17999999999999994
Epoch: 859, Loss: 0.18
Epoch: 860, Loss: 0.17999999999999994
Epoch: 861, Loss: 0.18
Epoch: 862, Loss: 0.17999999999999994
Epoch: 863, Loss: 0.18
Epoch: 864, Loss: 0.18
Epoch: 865, Loss: 0.17999999999999994
Epoch: 866, Loss: 0.18
Epoch: 867, Loss: 0.17999999999999994
Epoch: 868, Loss: 0.18
Epoch: 869, Loss: 0.17999999999999994
Epoch: 870, Loss: 0.18
Epoch: 871, Loss: 0.17999999999999994
Epoch: 872, Loss: 0.18
^CTraceback (most recent call last):
  File "/Users/kyleuphus/docs/ml-library/scratch/train_loop.py", line 17, in <module>
    ypred = [model(x)[0] for x in xs]
             ~~~~~^^^
  File "/Users/kyleuphus/docs/ml-library/src/mygrad/nn/layers.py", line 49, in __call__
    x = layer(x)
  File "/Users/kyleuphus/docs/ml-library/src/mygrad/nn/layers.py", line 35, in __call__
    outputs = [n(x) for n in self.neurons]
               ~^^^
  File "/Users/kyleuphus/docs/ml-library/src/mygrad/nn/layers.py", line 17, in __call__
    activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
  File "/Users/kyleuphus/docs/ml-library/src/mygrad/engine.py", line 15, in __add__
    out = Value(addition, (self, other))
  File "/Users/kyleuphus/docs/ml-library/src/mygrad/engine.py", line 5, in __init__
    def __init__(self, data, _children=(), _op=""):
    
KeyboardInterrupt

(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library %  python3 scratch/train_loop.py
Epoch: 0, Loss: 9.964193015873882
Epoch: 1, Loss: 4.788613913209864
Epoch: 2, Loss: 2.840719229300594
Epoch: 3, Loss: 1.618590719429688
Epoch: 4, Loss: 0.8733927700480474
Epoch: 5, Loss: 0.5064665554190291
Epoch: 6, Loss: 0.3275925247863034
Epoch: 7, Loss: 0.24680910469011158
Epoch: 8, Loss: 0.20985367218166706
Epoch: 9, Loss: 0.1933858996959095
Epoch: 10, Loss: 0.18597471270274354
Epoch: 11, Loss: 0.18267239091565265
Epoch: 12, Loss: 0.18119317519172246
Epoch: 13, Loss: 0.1805332912054317
Epoch: 14, Loss: 0.1802381756278381
Epoch: 15, Loss: 0.18010642485922318
Epoch: 16, Loss: 0.18004753862465459
Epoch: 17, Loss: 0.18002123963427596
Epoch: 18, Loss: 0.18000948821947552
Epoch: 19, Loss: 0.1800042390260364
Epoch: 20, Loss: 0.18000189373677922
Epoch: 21, Loss: 0.18000084604363498
Epoch: 22, Loss: 0.18000037796670176
Epoch: 23, Loss: 0.18000016885857728
Epoch: 24, Loss: 0.1800000754374867
Epoch: 25, Loss: 0.18000003370196646
Epoch: 26, Loss: 0.18000001505638977
Epoch: 27, Loss: 0.18000000672648583
Epoch: 28, Loss: 0.18000000300506958
Epoch: 29, Loss: 0.18000000134252236
Epoch: 30, Loss: 0.18000000059977456
Epoch: 31, Loss: 0.18000000026795065
Epoch: 32, Loss: 0.18000000011970735
Epoch: 33, Loss: 0.18000000005347971
Epoch: 34, Loss: 0.18000000002389183
Epoch: 35, Loss: 0.1800000000106733
Epoch: 36, Loss: 0.18000000000476868
Epoch: 37, Loss: 0.18000000000213107
Epoch: 38, Loss: 0.1800000000009519
Epoch: 39, Loss: 0.1800000000004251
Epoch: 40, Loss: 0.18000000000018984
Epoch: 41, Loss: 0.18000000000008523
Epoch: 42, Loss: 0.18000000000003827
Epoch: 43, Loss: 0.18000000000001679
Epoch: 44, Loss: 0.1800000000000075
Epoch: 45, Loss: 0.1800000000000041
Epoch: 46, Loss: 0.18000000000000177
Epoch: 47, Loss: 0.18000000000000035
Epoch: 48, Loss: 0.18000000000000027
Epoch: 49, Loss: 0.17999999999999974
Epoch: 50, Loss: 0.1799999999999999
Epoch: 51, Loss: 0.1799999999999997
Epoch: 52, Loss: 0.18000000000000038
Epoch: 53, Loss: 0.18000000000000035
Epoch: 54, Loss: 0.1800000000000001
Epoch: 55, Loss: 0.1800000000000001
Epoch: 56, Loss: 0.17999999999999952
Epoch: 57, Loss: 0.17999999999999983
Epoch: 58, Loss: 0.18
Epoch: 59, Loss: 0.1800000000000002
Epoch: 60, Loss: 0.17999999999999994
Epoch: 61, Loss: 0.1799999999999996
Epoch: 62, Loss: 0.1800000000000002
Epoch: 63, Loss: 0.17999999999999947
Epoch: 64, Loss: 0.1799999999999998
Epoch: 65, Loss: 0.17999999999999952
Epoch: 66, Loss: 0.18
Epoch: 67, Loss: 0.18000000000000033
Epoch: 68, Loss: 0.17999999999999994
Epoch: 69, Loss: 0.17999999999999988
Epoch: 70, Loss: 0.18000000000000022
Epoch: 71, Loss: 0.17999999999999997
Epoch: 72, Loss: 0.17999999999999997
Epoch: 73, Loss: 0.17999999999999966
Epoch: 74, Loss: 0.17999999999999905
Epoch: 75, Loss: 0.18000000000000044
Epoch: 76, Loss: 0.1800000000000002
Epoch: 77, Loss: 0.18000000000000027
Epoch: 78, Loss: 0.17999999999999966
Epoch: 79, Loss: 0.18000000000000038
Epoch: 80, Loss: 0.18000000000000024
Epoch: 81, Loss: 0.17999999999999933
Epoch: 82, Loss: 0.18000000000000033
Epoch: 83, Loss: 0.18000000000000044
Epoch: 84, Loss: 0.17999999999999988
Epoch: 85, Loss: 0.17999999999999988
Epoch: 86, Loss: 0.17999999999999997
Epoch: 87, Loss: 0.18000000000000038
Epoch: 88, Loss: 0.18
Epoch: 89, Loss: 0.18000000000000005
Epoch: 90, Loss: 0.1800000000000001
Epoch: 91, Loss: 0.18000000000000016
Epoch: 92, Loss: 0.18000000000000016
Epoch: 93, Loss: 0.18000000000000016
Epoch: 94, Loss: 0.18000000000000016
Epoch: 95, Loss: 0.18000000000000016
Epoch: 96, Loss: 0.18000000000000016
Epoch: 97, Loss: 0.18000000000000016
Epoch: 98, Loss: 0.18000000000000016
Epoch: 99, Loss: 0.18000000000000016
Epoch: 100, Loss: 0.18000000000000016
Epoch: 101, Loss: 0.18000000000000016
Epoch: 102, Loss: 0.18000000000000016
Epoch: 103, Loss: 0.18000000000000016
Epoch: 104, Loss: 0.18000000000000016
Epoch: 105, Loss: 0.1800000000000001
Epoch: 106, Loss: 0.1800000000000001
Epoch: 107, Loss: 0.1800000000000001
Epoch: 108, Loss: 0.1800000000000001
Epoch: 109, Loss: 0.1800000000000001
Epoch: 110, Loss: 0.1800000000000001
Epoch: 111, Loss: 0.1800000000000001
Epoch: 112, Loss: 0.1800000000000001
Epoch: 113, Loss: 0.1800000000000001
Epoch: 114, Loss: 0.1800000000000001
Epoch: 115, Loss: 0.1800000000000001
Epoch: 116, Loss: 0.1800000000000001
Epoch: 117, Loss: 0.1800000000000001
Epoch: 118, Loss: 0.1800000000000001
Epoch: 119, Loss: 0.1800000000000001
Epoch: 120, Loss: 0.1800000000000001
Epoch: 121, Loss: 0.1800000000000001
Epoch: 122, Loss: 0.1800000000000001
Epoch: 123, Loss: 0.1800000000000001
Epoch: 124, Loss: 0.1800000000000001
Epoch: 125, Loss: 0.18000000000000005
Epoch: 126, Loss: 0.18000000000000005
Epoch: 127, Loss: 0.18000000000000005
Epoch: 128, Loss: 0.18000000000000005
Epoch: 129, Loss: 0.18000000000000005
Epoch: 130, Loss: 0.18000000000000005
Epoch: 131, Loss: 0.18000000000000005
Epoch: 132, Loss: 0.17999999999999994
Epoch: 133, Loss: 0.17999999999999994
Epoch: 134, Loss: 0.17999999999999994
Epoch: 135, Loss: 0.17999999999999994
Epoch: 136, Loss: 0.17999999999999994
Epoch: 137, Loss: 0.17999999999999994
Epoch: 138, Loss: 0.17999999999999994
Epoch: 139, Loss: 0.17999999999999994
Epoch: 140, Loss: 0.17999999999999994
Epoch: 141, Loss: 0.17999999999999994
Epoch: 142, Loss: 0.17999999999999994
Epoch: 143, Loss: 0.17999999999999994
Epoch: 144, Loss: 0.17999999999999994
Epoch: 145, Loss: 0.17999999999999994
Epoch: 146, Loss: 0.17999999999999994
Epoch: 147, Loss: 0.17999999999999994
Epoch: 148, Loss: 0.17999999999999994
Epoch: 149, Loss: 0.17999999999999994
Epoch: 150, Loss: 0.17999999999999994
Epoch: 151, Loss: 0.17999999999999994
Epoch: 152, Loss: 0.17999999999999994
Epoch: 153, Loss: 0.17999999999999994
Epoch: 154, Loss: 0.17999999999999994
Epoch: 155, Loss: 0.17999999999999994
Epoch: 156, Loss: 0.17999999999999994
Epoch: 157, Loss: 0.17999999999999994
Epoch: 158, Loss: 0.17999999999999994
Epoch: 159, Loss: 0.17999999999999994
Epoch: 160, Loss: 0.17999999999999994
Epoch: 161, Loss: 0.17999999999999994
Epoch: 162, Loss: 0.17999999999999994
Epoch: 163, Loss: 0.17999999999999994
Epoch: 164, Loss: 0.17999999999999994
Epoch: 165, Loss: 0.17999999999999994
Epoch: 166, Loss: 0.17999999999999994
Epoch: 167, Loss: 0.17999999999999994
Epoch: 168, Loss: 0.17999999999999994
Epoch: 169, Loss: 0.17999999999999994
Epoch: 170, Loss: 0.17999999999999994
Epoch: 171, Loss: 0.17999999999999994
Epoch: 172, Loss: 0.17999999999999994
Epoch: 173, Loss: 0.17999999999999994
Epoch: 174, Loss: 0.17999999999999994
Epoch: 175, Loss: 0.17999999999999994
Epoch: 176, Loss: 0.17999999999999994
Epoch: 177, Loss: 0.17999999999999994
Epoch: 178, Loss: 0.17999999999999994
Epoch: 179, Loss: 0.17999999999999994
Epoch: 180, Loss: 0.17999999999999994
Epoch: 181, Loss: 0.17999999999999994
Epoch: 182, Loss: 0.17999999999999994
Epoch: 183, Loss: 0.17999999999999994
Epoch: 184, Loss: 0.17999999999999994
Epoch: 185, Loss: 0.17999999999999994
Epoch: 186, Loss: 0.17999999999999994
Epoch: 187, Loss: 0.17999999999999994
Epoch: 188, Loss: 0.17999999999999994
Epoch: 189, Loss: 0.17999999999999994
Epoch: 190, Loss: 0.17999999999999994
Epoch: 191, Loss: 0.17999999999999994
Epoch: 192, Loss: 0.17999999999999994
Epoch: 193, Loss: 0.17999999999999994
Epoch: 194, Loss: 0.17999999999999994
Epoch: 195, Loss: 0.17999999999999994
Epoch: 196, Loss: 0.17999999999999994
Epoch: 197, Loss: 0.17999999999999994
Epoch: 198, Loss: 0.17999999999999994
Epoch: 199, Loss: 0.17999999999999994
Epoch: 200, Loss: 0.17999999999999994
Epoch: 201, Loss: 0.17999999999999994
Epoch: 202, Loss: 0.17999999999999994
Epoch: 203, Loss: 0.17999999999999994
Epoch: 204, Loss: 0.17999999999999994
Epoch: 205, Loss: 0.17999999999999994
Epoch: 206, Loss: 0.17999999999999994
Epoch: 207, Loss: 0.17999999999999994
Epoch: 208, Loss: 0.17999999999999994
Epoch: 209, Loss: 0.17999999999999994
Epoch: 210, Loss: 0.17999999999999994
Epoch: 211, Loss: 0.17999999999999994
Epoch: 212, Loss: 0.17999999999999994
Epoch: 213, Loss: 0.17999999999999994
Epoch: 214, Loss: 0.17999999999999994
Epoch: 215, Loss: 0.17999999999999994
Epoch: 216, Loss: 0.17999999999999994
Epoch: 217, Loss: 0.17999999999999994
Epoch: 218, Loss: 0.17999999999999994
Epoch: 219, Loss: 0.17999999999999994
Epoch: 220, Loss: 0.17999999999999994
Epoch: 221, Loss: 0.17999999999999994
Epoch: 222, Loss: 0.17999999999999994
Epoch: 223, Loss: 0.17999999999999994
Epoch: 224, Loss: 0.17999999999999994
Epoch: 225, Loss: 0.17999999999999994
Epoch: 226, Loss: 0.17999999999999994
Epoch: 227, Loss: 0.17999999999999994
Epoch: 228, Loss: 0.17999999999999994
Epoch: 229, Loss: 0.17999999999999994
Epoch: 230, Loss: 0.17999999999999994
Epoch: 231, Loss: 0.17999999999999994
Epoch: 232, Loss: 0.17999999999999994
Epoch: 233, Loss: 0.17999999999999994
Epoch: 234, Loss: 0.17999999999999994
Epoch: 235, Loss: 0.17999999999999994
Epoch: 236, Loss: 0.17999999999999994
Epoch: 237, Loss: 0.17999999999999994
Epoch: 238, Loss: 0.17999999999999994
Epoch: 239, Loss: 0.17999999999999994
Epoch: 240, Loss: 0.17999999999999994
Epoch: 241, Loss: 0.17999999999999994
Epoch: 242, Loss: 0.17999999999999994
Epoch: 243, Loss: 0.17999999999999994
Epoch: 244, Loss: 0.17999999999999994
Epoch: 245, Loss: 0.17999999999999994
Epoch: 246, Loss: 0.17999999999999994
Epoch: 247, Loss: 0.17999999999999994
Epoch: 248, Loss: 0.17999999999999994
Epoch: 249, Loss: 0.17999999999999994
Epoch: 250, Loss: 0.17999999999999994
Epoch: 251, Loss: 0.17999999999999994
Epoch: 252, Loss: 0.17999999999999994
Epoch: 253, Loss: 0.17999999999999994
Epoch: 254, Loss: 0.17999999999999994
Epoch: 255, Loss: 0.17999999999999994
Epoch: 256, Loss: 0.17999999999999994
Epoch: 257, Loss: 0.17999999999999994
Epoch: 258, Loss: 0.17999999999999994
Epoch: 259, Loss: 0.17999999999999994
Epoch: 260, Loss: 0.17999999999999994
Epoch: 261, Loss: 0.17999999999999994
Epoch: 262, Loss: 0.17999999999999994
Epoch: 263, Loss: 0.17999999999999994
Epoch: 264, Loss: 0.17999999999999994
Epoch: 265, Loss: 0.17999999999999994
Epoch: 266, Loss: 0.17999999999999994
Epoch: 267, Loss: 0.17999999999999994
Epoch: 268, Loss: 0.17999999999999994
Epoch: 269, Loss: 0.17999999999999994
Epoch: 270, Loss: 0.17999999999999994
Epoch: 271, Loss: 0.17999999999999994
Epoch: 272, Loss: 0.17999999999999994
Epoch: 273, Loss: 0.17999999999999994
Epoch: 274, Loss: 0.17999999999999994
Epoch: 275, Loss: 0.17999999999999994
Epoch: 276, Loss: 0.17999999999999994
Epoch: 277, Loss: 0.17999999999999994
Epoch: 278, Loss: 0.17999999999999977
Epoch: 279, Loss: 0.17999999999999977
Epoch: 280, Loss: 0.17999999999999977
Epoch: 281, Loss: 0.17999999999999977
Epoch: 282, Loss: 0.17999999999999977
Epoch: 283, Loss: 0.17999999999999977
Epoch: 284, Loss: 0.17999999999999977
Epoch: 285, Loss: 0.17999999999999977
Epoch: 286, Loss: 0.17999999999999977
Epoch: 287, Loss: 0.17999999999999977
Epoch: 288, Loss: 0.17999999999999977
Epoch: 289, Loss: 0.17999999999999977
Epoch: 290, Loss: 0.17999999999999977
Epoch: 291, Loss: 0.17999999999999977
Epoch: 292, Loss: 0.17999999999999977
Epoch: 293, Loss: 0.17999999999999977
Epoch: 294, Loss: 0.17999999999999977
Epoch: 295, Loss: 0.17999999999999977
Epoch: 296, Loss: 0.17999999999999977
Epoch: 297, Loss: 0.17999999999999977
Epoch: 298, Loss: 0.17999999999999977
Epoch: 299, Loss: 0.17999999999999977
Epoch: 300, Loss: 0.17999999999999977
Epoch: 301, Loss: 0.17999999999999977
Epoch: 302, Loss: 0.17999999999999994
Epoch: 303, Loss: 0.17999999999999977
Epoch: 304, Loss: 0.17999999999999994
Epoch: 305, Loss: 0.17999999999999977
Epoch: 306, Loss: 0.17999999999999994
Epoch: 307, Loss: 0.17999999999999977
Epoch: 308, Loss: 0.17999999999999994
Epoch: 309, Loss: 0.17999999999999977
Epoch: 310, Loss: 0.17999999999999994
Epoch: 311, Loss: 0.17999999999999977
Epoch: 312, Loss: 0.17999999999999994
Epoch: 313, Loss: 0.17999999999999977
Epoch: 314, Loss: 0.17999999999999994
Epoch: 315, Loss: 0.17999999999999977
Epoch: 316, Loss: 0.17999999999999994
Epoch: 317, Loss: 0.17999999999999977
Epoch: 318, Loss: 0.17999999999999994
Epoch: 319, Loss: 0.17999999999999977
Epoch: 320, Loss: 0.17999999999999994
Epoch: 321, Loss: 0.17999999999999977
Epoch: 322, Loss: 0.17999999999999994
Epoch: 323, Loss: 0.17999999999999977
Epoch: 324, Loss: 0.17999999999999994
Epoch: 325, Loss: 0.17999999999999977
Epoch: 326, Loss: 0.17999999999999994
Epoch: 327, Loss: 0.17999999999999977
Epoch: 328, Loss: 0.17999999999999994
Epoch: 329, Loss: 0.17999999999999977
Epoch: 330, Loss: 0.17999999999999994
Epoch: 331, Loss: 0.17999999999999977
Epoch: 332, Loss: 0.17999999999999994
Epoch: 333, Loss: 0.17999999999999977
Epoch: 334, Loss: 0.17999999999999994
Epoch: 335, Loss: 0.17999999999999977
Epoch: 336, Loss: 0.17999999999999994
Epoch: 337, Loss: 0.17999999999999977
Epoch: 338, Loss: 0.17999999999999994
Epoch: 339, Loss: 0.17999999999999977
Epoch: 340, Loss: 0.17999999999999994
Epoch: 341, Loss: 0.17999999999999977
Epoch: 342, Loss: 0.17999999999999994
Epoch: 343, Loss: 0.17999999999999977
Epoch: 344, Loss: 0.17999999999999994
Epoch: 345, Loss: 0.17999999999999977
Epoch: 346, Loss: 0.17999999999999994
Epoch: 347, Loss: 0.17999999999999977
Epoch: 348, Loss: 0.17999999999999994
Epoch: 349, Loss: 0.17999999999999977
Epoch: 350, Loss: 0.17999999999999994
Epoch: 351, Loss: 0.17999999999999977
Epoch: 352, Loss: 0.17999999999999994
Epoch: 353, Loss: 0.17999999999999977
Epoch: 354, Loss: 0.17999999999999994
Epoch: 355, Loss: 0.17999999999999977
Epoch: 356, Loss: 0.17999999999999994
Epoch: 357, Loss: 0.17999999999999977
Epoch: 358, Loss: 0.17999999999999994
Epoch: 359, Loss: 0.17999999999999977
Epoch: 360, Loss: 0.17999999999999994
Epoch: 361, Loss: 0.17999999999999977
Epoch: 362, Loss: 0.17999999999999994
Epoch: 363, Loss: 0.17999999999999977
Epoch: 364, Loss: 0.17999999999999994
Epoch: 365, Loss: 0.17999999999999977
Epoch: 366, Loss: 0.17999999999999994
Epoch: 367, Loss: 0.17999999999999977
Epoch: 368, Loss: 0.17999999999999994
Epoch: 369, Loss: 0.17999999999999977
Epoch: 370, Loss: 0.17999999999999994
Epoch: 371, Loss: 0.17999999999999977
Epoch: 372, Loss: 0.17999999999999994
Epoch: 373, Loss: 0.17999999999999977
Epoch: 374, Loss: 0.17999999999999994
Epoch: 375, Loss: 0.17999999999999977
Epoch: 376, Loss: 0.17999999999999994
Epoch: 377, Loss: 0.17999999999999977
Epoch: 378, Loss: 0.17999999999999994
Epoch: 379, Loss: 0.17999999999999977
Epoch: 380, Loss: 0.17999999999999994
Epoch: 381, Loss: 0.17999999999999977
Epoch: 382, Loss: 0.17999999999999994
Epoch: 383, Loss: 0.17999999999999977
Epoch: 384, Loss: 0.17999999999999994
Epoch: 385, Loss: 0.17999999999999977
Epoch: 386, Loss: 0.17999999999999994
Epoch: 387, Loss: 0.17999999999999977
Epoch: 388, Loss: 0.17999999999999994
Epoch: 389, Loss: 0.17999999999999977
Epoch: 390, Loss: 0.17999999999999994
Epoch: 391, Loss: 0.17999999999999977
Epoch: 392, Loss: 0.17999999999999994
Epoch: 393, Loss: 0.17999999999999977
Epoch: 394, Loss: 0.17999999999999994
Epoch: 395, Loss: 0.17999999999999977
Epoch: 396, Loss: 0.17999999999999994
Epoch: 397, Loss: 0.17999999999999977
Epoch: 398, Loss: 0.17999999999999994
Epoch: 399, Loss: 0.17999999999999977
Epoch: 400, Loss: 0.17999999999999994
Epoch: 401, Loss: 0.17999999999999977
Epoch: 402, Loss: 0.17999999999999994
Epoch: 403, Loss: 0.17999999999999977
Epoch: 404, Loss: 0.17999999999999994
Epoch: 405, Loss: 0.17999999999999977
Epoch: 406, Loss: 0.17999999999999994
Epoch: 407, Loss: 0.17999999999999977
Epoch: 408, Loss: 0.17999999999999994
Epoch: 409, Loss: 0.17999999999999977
Epoch: 410, Loss: 0.17999999999999994
Epoch: 411, Loss: 0.17999999999999977
Epoch: 412, Loss: 0.17999999999999994
Epoch: 413, Loss: 0.17999999999999977
Epoch: 414, Loss: 0.17999999999999994
Epoch: 415, Loss: 0.17999999999999977
Epoch: 416, Loss: 0.17999999999999994
Epoch: 417, Loss: 0.17999999999999977
Epoch: 418, Loss: 0.17999999999999994
Epoch: 419, Loss: 0.17999999999999977
Epoch: 420, Loss: 0.17999999999999994
Epoch: 421, Loss: 0.17999999999999977
Epoch: 422, Loss: 0.17999999999999994
Epoch: 423, Loss: 0.17999999999999977
Epoch: 424, Loss: 0.17999999999999994
Epoch: 425, Loss: 0.17999999999999977
Epoch: 426, Loss: 0.17999999999999994
Epoch: 427, Loss: 0.17999999999999977
Epoch: 428, Loss: 0.17999999999999994
Epoch: 429, Loss: 0.17999999999999977
Epoch: 430, Loss: 0.17999999999999994
Epoch: 431, Loss: 0.17999999999999977
Epoch: 432, Loss: 0.17999999999999994
Epoch: 433, Loss: 0.17999999999999977
Epoch: 434, Loss: 0.17999999999999994
Epoch: 435, Loss: 0.17999999999999977
Epoch: 436, Loss: 0.17999999999999994
Epoch: 437, Loss: 0.17999999999999977
Epoch: 438, Loss: 0.17999999999999994
Epoch: 439, Loss: 0.17999999999999977
Epoch: 440, Loss: 0.17999999999999994
Epoch: 441, Loss: 0.17999999999999977
Epoch: 442, Loss: 0.17999999999999994
Epoch: 443, Loss: 0.17999999999999977
Epoch: 444, Loss: 0.17999999999999994
Epoch: 445, Loss: 0.17999999999999977
Epoch: 446, Loss: 0.17999999999999994
Epoch: 447, Loss: 0.17999999999999977
Epoch: 448, Loss: 0.17999999999999994
Epoch: 449, Loss: 0.17999999999999977
Epoch: 450, Loss: 0.17999999999999994
Epoch: 451, Loss: 0.17999999999999977
Epoch: 452, Loss: 0.17999999999999994
Epoch: 453, Loss: 0.17999999999999977
Epoch: 454, Loss: 0.17999999999999994
Epoch: 455, Loss: 0.17999999999999977
Epoch: 456, Loss: 0.17999999999999994
Epoch: 457, Loss: 0.17999999999999977
Epoch: 458, Loss: 0.17999999999999994
Epoch: 459, Loss: 0.17999999999999977
Epoch: 460, Loss: 0.17999999999999994
Epoch: 461, Loss: 0.17999999999999977
Epoch: 462, Loss: 0.17999999999999994
Epoch: 463, Loss: 0.17999999999999977
Epoch: 464, Loss: 0.17999999999999994
Epoch: 465, Loss: 0.17999999999999977
Epoch: 466, Loss: 0.17999999999999994
Epoch: 467, Loss: 0.17999999999999977
Epoch: 468, Loss: 0.17999999999999994
Epoch: 469, Loss: 0.17999999999999977
Epoch: 470, Loss: 0.17999999999999994
Epoch: 471, Loss: 0.17999999999999977
Epoch: 472, Loss: 0.17999999999999994
Epoch: 473, Loss: 0.17999999999999977
Epoch: 474, Loss: 0.17999999999999994
Epoch: 475, Loss: 0.17999999999999977
Epoch: 476, Loss: 0.17999999999999994
Epoch: 477, Loss: 0.17999999999999977
Epoch: 478, Loss: 0.17999999999999994
Epoch: 479, Loss: 0.17999999999999977
Epoch: 480, Loss: 0.17999999999999994
Epoch: 481, Loss: 0.17999999999999977
Epoch: 482, Loss: 0.17999999999999994
Epoch: 483, Loss: 0.17999999999999977
Epoch: 484, Loss: 0.17999999999999994
Epoch: 485, Loss: 0.17999999999999977
Epoch: 486, Loss: 0.17999999999999994
Epoch: 487, Loss: 0.17999999999999977
Epoch: 488, Loss: 0.17999999999999994
Epoch: 489, Loss: 0.17999999999999977
Epoch: 490, Loss: 0.17999999999999994
Epoch: 491, Loss: 0.17999999999999977
Epoch: 492, Loss: 0.17999999999999994
Epoch: 493, Loss: 0.17999999999999977
Epoch: 494, Loss: 0.17999999999999994
Epoch: 495, Loss: 0.17999999999999977
Epoch: 496, Loss: 0.17999999999999994
Epoch: 497, Loss: 0.17999999999999977
Epoch: 498, Loss: 0.17999999999999994
Epoch: 499, Loss: 0.17999999999999977
Epoch: 500, Loss: 0.17999999999999994
Epoch: 501, Loss: 0.17999999999999977
Epoch: 502, Loss: 0.17999999999999994
Epoch: 503, Loss: 0.17999999999999977
Epoch: 504, Loss: 0.17999999999999994
Epoch: 505, Loss: 0.17999999999999977
Epoch: 506, Loss: 0.17999999999999994
Epoch: 507, Loss: 0.17999999999999977
Epoch: 508, Loss: 0.17999999999999994
Epoch: 509, Loss: 0.17999999999999977
Epoch: 510, Loss: 0.17999999999999994
Epoch: 511, Loss: 0.17999999999999977
Epoch: 512, Loss: 0.17999999999999994
Epoch: 513, Loss: 0.17999999999999977
Epoch: 514, Loss: 0.17999999999999994
Epoch: 515, Loss: 0.17999999999999977
Epoch: 516, Loss: 0.17999999999999994
Epoch: 517, Loss: 0.17999999999999977
Epoch: 518, Loss: 0.17999999999999994
Epoch: 519, Loss: 0.17999999999999977
Epoch: 520, Loss: 0.17999999999999994
Epoch: 521, Loss: 0.17999999999999977
Epoch: 522, Loss: 0.17999999999999994
Epoch: 523, Loss: 0.17999999999999977
Epoch: 524, Loss: 0.17999999999999994
Epoch: 525, Loss: 0.17999999999999977
Epoch: 526, Loss: 0.17999999999999994
Epoch: 527, Loss: 0.17999999999999977
Epoch: 528, Loss: 0.17999999999999994
Epoch: 529, Loss: 0.17999999999999977
Epoch: 530, Loss: 0.17999999999999994
Epoch: 531, Loss: 0.17999999999999977
Epoch: 532, Loss: 0.17999999999999994
Epoch: 533, Loss: 0.17999999999999977
Epoch: 534, Loss: 0.17999999999999994
Epoch: 535, Loss: 0.17999999999999977
Epoch: 536, Loss: 0.17999999999999994
Epoch: 537, Loss: 0.17999999999999977
Epoch: 538, Loss: 0.17999999999999994
Epoch: 539, Loss: 0.17999999999999977
Epoch: 540, Loss: 0.17999999999999994
Epoch: 541, Loss: 0.17999999999999977
Epoch: 542, Loss: 0.17999999999999994
Epoch: 543, Loss: 0.17999999999999977
Epoch: 544, Loss: 0.17999999999999994
Epoch: 545, Loss: 0.17999999999999977
Epoch: 546, Loss: 0.17999999999999994
Epoch: 547, Loss: 0.17999999999999977
Epoch: 548, Loss: 0.17999999999999994
Epoch: 549, Loss: 0.17999999999999977
Epoch: 550, Loss: 0.17999999999999994
Epoch: 551, Loss: 0.17999999999999977
Epoch: 552, Loss: 0.17999999999999994
Epoch: 553, Loss: 0.17999999999999977
Epoch: 554, Loss: 0.17999999999999994
Epoch: 555, Loss: 0.17999999999999977
Epoch: 556, Loss: 0.17999999999999994
Epoch: 557, Loss: 0.17999999999999977
Epoch: 558, Loss: 0.17999999999999994
Epoch: 559, Loss: 0.17999999999999977
Epoch: 560, Loss: 0.17999999999999994
Epoch: 561, Loss: 0.17999999999999977
Epoch: 562, Loss: 0.17999999999999994
Epoch: 563, Loss: 0.17999999999999977
Epoch: 564, Loss: 0.17999999999999994
Epoch: 565, Loss: 0.17999999999999977
Epoch: 566, Loss: 0.17999999999999994
Epoch: 567, Loss: 0.17999999999999977
Epoch: 568, Loss: 0.17999999999999994
Epoch: 569, Loss: 0.17999999999999977
Epoch: 570, Loss: 0.17999999999999994
Epoch: 571, Loss: 0.17999999999999977
Epoch: 572, Loss: 0.17999999999999994
Epoch: 573, Loss: 0.17999999999999977
Epoch: 574, Loss: 0.17999999999999994
Epoch: 575, Loss: 0.17999999999999977
Epoch: 576, Loss: 0.17999999999999994
Epoch: 577, Loss: 0.17999999999999977
Epoch: 578, Loss: 0.17999999999999994
Epoch: 579, Loss: 0.17999999999999977
Epoch: 580, Loss: 0.17999999999999994
Epoch: 581, Loss: 0.17999999999999977
Epoch: 582, Loss: 0.17999999999999994
Epoch: 583, Loss: 0.17999999999999977
Epoch: 584, Loss: 0.17999999999999994
Epoch: 585, Loss: 0.17999999999999977
Epoch: 586, Loss: 0.17999999999999994
Epoch: 587, Loss: 0.17999999999999977
Epoch: 588, Loss: 0.17999999999999994
Epoch: 589, Loss: 0.17999999999999977
Epoch: 590, Loss: 0.17999999999999994
Epoch: 591, Loss: 0.17999999999999977
Epoch: 592, Loss: 0.17999999999999994
Epoch: 593, Loss: 0.17999999999999977
Epoch: 594, Loss: 0.17999999999999994
Epoch: 595, Loss: 0.17999999999999977
Epoch: 596, Loss: 0.17999999999999994
Epoch: 597, Loss: 0.17999999999999977
Epoch: 598, Loss: 0.17999999999999994
Epoch: 599, Loss: 0.17999999999999977
Epoch: 600, Loss: 0.17999999999999994
Epoch: 601, Loss: 0.17999999999999977
Epoch: 602, Loss: 0.17999999999999994
Epoch: 603, Loss: 0.17999999999999977
Epoch: 604, Loss: 0.17999999999999994
Epoch: 605, Loss: 0.17999999999999977
Epoch: 606, Loss: 0.17999999999999994
Epoch: 607, Loss: 0.17999999999999977
Epoch: 608, Loss: 0.17999999999999994
Epoch: 609, Loss: 0.17999999999999977
Epoch: 610, Loss: 0.17999999999999994
Epoch: 611, Loss: 0.17999999999999977
Epoch: 612, Loss: 0.17999999999999994
Epoch: 613, Loss: 0.17999999999999977
Epoch: 614, Loss: 0.17999999999999994
Epoch: 615, Loss: 0.17999999999999977
Epoch: 616, Loss: 0.17999999999999994
Epoch: 617, Loss: 0.17999999999999977
Epoch: 618, Loss: 0.17999999999999994
Epoch: 619, Loss: 0.17999999999999977
Epoch: 620, Loss: 0.17999999999999994
Epoch: 621, Loss: 0.17999999999999977
Epoch: 622, Loss: 0.17999999999999994
Epoch: 623, Loss: 0.17999999999999977
Epoch: 624, Loss: 0.17999999999999994
Epoch: 625, Loss: 0.17999999999999977
Epoch: 626, Loss: 0.17999999999999994
Epoch: 627, Loss: 0.17999999999999977
Epoch: 628, Loss: 0.17999999999999994
Epoch: 629, Loss: 0.17999999999999977
Epoch: 630, Loss: 0.17999999999999994
Epoch: 631, Loss: 0.17999999999999977
Epoch: 632, Loss: 0.17999999999999994
Epoch: 633, Loss: 0.17999999999999977
Epoch: 634, Loss: 0.17999999999999994
Epoch: 635, Loss: 0.17999999999999977
Epoch: 636, Loss: 0.17999999999999994
Epoch: 637, Loss: 0.17999999999999977
Epoch: 638, Loss: 0.17999999999999994
Epoch: 639, Loss: 0.17999999999999977
Epoch: 640, Loss: 0.17999999999999994
Epoch: 641, Loss: 0.17999999999999977
Epoch: 642, Loss: 0.17999999999999994
Epoch: 643, Loss: 0.17999999999999977
Epoch: 644, Loss: 0.17999999999999994
Epoch: 645, Loss: 0.17999999999999977
Epoch: 646, Loss: 0.17999999999999994
Epoch: 647, Loss: 0.17999999999999977
Epoch: 648, Loss: 0.17999999999999994
Epoch: 649, Loss: 0.17999999999999977
Epoch: 650, Loss: 0.17999999999999994
Epoch: 651, Loss: 0.17999999999999977
Epoch: 652, Loss: 0.17999999999999994
Epoch: 653, Loss: 0.17999999999999977
Epoch: 654, Loss: 0.17999999999999994
Epoch: 655, Loss: 0.17999999999999977
Epoch: 656, Loss: 0.17999999999999994
Epoch: 657, Loss: 0.17999999999999977
Epoch: 658, Loss: 0.17999999999999994
Epoch: 659, Loss: 0.17999999999999977
Epoch: 660, Loss: 0.17999999999999994
Epoch: 661, Loss: 0.17999999999999977
Epoch: 662, Loss: 0.17999999999999994
Epoch: 663, Loss: 0.17999999999999977
Epoch: 664, Loss: 0.17999999999999994
Epoch: 665, Loss: 0.17999999999999988
Epoch: 666, Loss: 0.18000000000000002
Epoch: 667, Loss: 0.17999999999999988
Epoch: 668, Loss: 0.17999999999999988
Epoch: 669, Loss: 0.17999999999999988
Epoch: 670, Loss: 0.17999999999999977
Epoch: 671, Loss: 0.17999999999999994
Epoch: 672, Loss: 0.17999999999999988
Epoch: 673, Loss: 0.17999999999999994
Epoch: 674, Loss: 0.17999999999999988
Epoch: 675, Loss: 0.17999999999999994
Epoch: 676, Loss: 0.17999999999999988
Epoch: 677, Loss: 0.17999999999999994
Epoch: 678, Loss: 0.17999999999999988
Epoch: 679, Loss: 0.17999999999999977
Epoch: 680, Loss: 0.17999999999999994
Epoch: 681, Loss: 0.17999999999999988
Epoch: 682, Loss: 0.18
Epoch: 683, Loss: 0.17999999999999994
Epoch: 684, Loss: 0.18
Epoch: 685, Loss: 0.1800000000000001
Epoch: 686, Loss: 0.18
Epoch: 687, Loss: 0.18000000000000005
Epoch: 688, Loss: 0.17999999999999988
Epoch: 689, Loss: 0.18000000000000005
Epoch: 690, Loss: 0.17999999999999988
Epoch: 691, Loss: 0.18000000000000005
Epoch: 692, Loss: 0.18000000000000005
Epoch: 693, Loss: 0.18
Epoch: 694, Loss: 0.18000000000000005
Epoch: 695, Loss: 0.17999999999999988
Epoch: 696, Loss: 0.18000000000000005
Epoch: 697, Loss: 0.17999999999999988
Epoch: 698, Loss: 0.18000000000000005
Epoch: 699, Loss: 0.17999999999999988
Epoch: 700, Loss: 0.18000000000000005
Epoch: 701, Loss: 0.17999999999999994
Epoch: 702, Loss: 0.17999999999999988
Epoch: 703, Loss: 0.17999999999999988
Epoch: 704, Loss: 0.18000000000000005
Epoch: 705, Loss: 0.17999999999999977
Epoch: 706, Loss: 0.17999999999999994
Epoch: 707, Loss: 0.17999999999999977
Epoch: 708, Loss: 0.17999999999999994
Epoch: 709, Loss: 0.17999999999999977
Epoch: 710, Loss: 0.17999999999999994
Epoch: 711, Loss: 0.17999999999999977
Epoch: 712, Loss: 0.17999999999999977
Epoch: 713, Loss: 0.17999999999999994
Epoch: 714, Loss: 0.17999999999999977
Epoch: 715, Loss: 0.17999999999999994
Epoch: 716, Loss: 0.17999999999999977
Epoch: 717, Loss: 0.17999999999999977
Epoch: 718, Loss: 0.17999999999999994
Epoch: 719, Loss: 0.17999999999999977
Epoch: 720, Loss: 0.17999999999999994
Epoch: 721, Loss: 0.17999999999999977
Epoch: 722, Loss: 0.17999999999999977
Epoch: 723, Loss: 0.17999999999999994
Epoch: 724, Loss: 0.17999999999999977
Epoch: 725, Loss: 0.17999999999999994
Epoch: 726, Loss: 0.17999999999999977
Epoch: 727, Loss: 0.17999999999999994
Epoch: 728, Loss: 0.17999999999999977
Epoch: 729, Loss: 0.17999999999999977
Epoch: 730, Loss: 0.17999999999999994
Epoch: 731, Loss: 0.17999999999999977
Epoch: 732, Loss: 0.17999999999999994
Epoch: 733, Loss: 0.17999999999999977
Epoch: 734, Loss: 0.17999999999999977
Epoch: 735, Loss: 0.17999999999999994
Epoch: 736, Loss: 0.17999999999999977
Epoch: 737, Loss: 0.17999999999999994
Epoch: 738, Loss: 0.17999999999999977
Epoch: 739, Loss: 0.17999999999999977
Epoch: 740, Loss: 0.17999999999999994
Epoch: 741, Loss: 0.17999999999999977
Epoch: 742, Loss: 0.17999999999999994
Epoch: 743, Loss: 0.17999999999999977
Epoch: 744, Loss: 0.17999999999999977
Epoch: 745, Loss: 0.17999999999999994
Epoch: 746, Loss: 0.17999999999999977
Epoch: 747, Loss: 0.17999999999999994
Epoch: 748, Loss: 0.17999999999999977
Epoch: 749, Loss: 0.17999999999999994
Epoch: 750, Loss: 0.17999999999999977
Epoch: 751, Loss: 0.17999999999999977
Epoch: 752, Loss: 0.17999999999999994
Epoch: 753, Loss: 0.17999999999999977
Epoch: 754, Loss: 0.17999999999999994
Epoch: 755, Loss: 0.17999999999999977
Epoch: 756, Loss: 0.17999999999999977
Epoch: 757, Loss: 0.17999999999999994
Epoch: 758, Loss: 0.17999999999999977
Epoch: 759, Loss: 0.17999999999999994
Epoch: 760, Loss: 0.17999999999999977
Epoch: 761, Loss: 0.17999999999999994
Epoch: 762, Loss: 0.17999999999999977
Epoch: 763, Loss: 0.17999999999999977
Epoch: 764, Loss: 0.17999999999999994
Epoch: 765, Loss: 0.17999999999999977
Epoch: 766, Loss: 0.17999999999999994
Epoch: 767, Loss: 0.17999999999999977
Epoch: 768, Loss: 0.17999999999999977
Epoch: 769, Loss: 0.17999999999999994
Epoch: 770, Loss: 0.17999999999999977
Epoch: 771, Loss: 0.17999999999999994
Epoch: 772, Loss: 0.17999999999999977
Epoch: 773, Loss: 0.17999999999999994
Epoch: 774, Loss: 0.17999999999999977
Epoch: 775, Loss: 0.17999999999999977
Epoch: 776, Loss: 0.17999999999999994
Epoch: 777, Loss: 0.17999999999999977
Epoch: 778, Loss: 0.17999999999999994
Epoch: 779, Loss: 0.17999999999999977
Epoch: 780, Loss: 0.17999999999999977
Epoch: 781, Loss: 0.17999999999999994
Epoch: 782, Loss: 0.17999999999999977
Epoch: 783, Loss: 0.17999999999999994
Epoch: 784, Loss: 0.17999999999999977
Epoch: 785, Loss: 0.17999999999999977
Epoch: 786, Loss: 0.17999999999999994
Epoch: 787, Loss: 0.17999999999999977
Epoch: 788, Loss: 0.17999999999999994
Epoch: 789, Loss: 0.17999999999999977
Epoch: 790, Loss: 0.17999999999999994
Epoch: 791, Loss: 0.17999999999999977
Epoch: 792, Loss: 0.17999999999999977
Epoch: 793, Loss: 0.17999999999999994
Epoch: 794, Loss: 0.17999999999999977
Epoch: 795, Loss: 0.17999999999999994
Epoch: 796, Loss: 0.17999999999999977
Epoch: 797, Loss: 0.17999999999999994
Epoch: 798, Loss: 0.17999999999999977
Epoch: 799, Loss: 0.17999999999999977
Epoch: 800, Loss: 0.17999999999999994
Epoch: 801, Loss: 0.17999999999999977
Epoch: 802, Loss: 0.17999999999999994
Epoch: 803, Loss: 0.17999999999999977
Epoch: 804, Loss: 0.17999999999999977
Epoch: 805, Loss: 0.17999999999999994
Epoch: 806, Loss: 0.17999999999999977
Epoch: 807, Loss: 0.17999999999999994
Epoch: 808, Loss: 0.17999999999999977
Epoch: 809, Loss: 0.17999999999999977
Epoch: 810, Loss: 0.17999999999999994
Epoch: 811, Loss: 0.17999999999999977
Epoch: 812, Loss: 0.17999999999999994
Epoch: 813, Loss: 0.17999999999999977
Epoch: 814, Loss: 0.17999999999999994
Epoch: 815, Loss: 0.17999999999999977
Epoch: 816, Loss: 0.17999999999999977
Epoch: 817, Loss: 0.17999999999999994
Epoch: 818, Loss: 0.17999999999999977
Epoch: 819, Loss: 0.17999999999999994
Epoch: 820, Loss: 0.17999999999999977
Epoch: 821, Loss: 0.17999999999999977
Epoch: 822, Loss: 0.17999999999999994
Epoch: 823, Loss: 0.17999999999999977
Epoch: 824, Loss: 0.17999999999999994
Epoch: 825, Loss: 0.17999999999999977
Epoch: 826, Loss: 0.17999999999999994
Epoch: 827, Loss: 0.17999999999999977
Epoch: 828, Loss: 0.17999999999999977
Epoch: 829, Loss: 0.17999999999999994
Epoch: 830, Loss: 0.17999999999999977
Epoch: 831, Loss: 0.17999999999999994
Epoch: 832, Loss: 0.17999999999999977
Epoch: 833, Loss: 0.17999999999999977
Epoch: 834, Loss: 0.17999999999999994
Epoch: 835, Loss: 0.17999999999999977
Epoch: 836, Loss: 0.17999999999999977
Epoch: 837, Loss: 0.17999999999999994
Epoch: 838, Loss: 0.17999999999999977
Epoch: 839, Loss: 0.17999999999999994
Epoch: 840, Loss: 0.17999999999999977
Epoch: 841, Loss: 0.17999999999999994
Epoch: 842, Loss: 0.17999999999999977
Epoch: 843, Loss: 0.17999999999999977
Epoch: 844, Loss: 0.17999999999999994
Epoch: 845, Loss: 0.17999999999999977
Epoch: 846, Loss: 0.17999999999999994
Epoch: 847, Loss: 0.17999999999999977
Epoch: 848, Loss: 0.17999999999999994
Epoch: 849, Loss: 0.17999999999999977
Epoch: 850, Loss: 0.17999999999999977
Epoch: 851, Loss: 0.17999999999999994
Epoch: 852, Loss: 0.17999999999999977
Epoch: 853, Loss: 0.17999999999999977
Epoch: 854, Loss: 0.17999999999999994
Epoch: 855, Loss: 0.17999999999999977
Epoch: 856, Loss: 0.17999999999999994
Epoch: 857, Loss: 0.17999999999999977
Epoch: 858, Loss: 0.17999999999999977
Epoch: 859, Loss: 0.17999999999999994
Epoch: 860, Loss: 0.17999999999999977
Epoch: 861, Loss: 0.17999999999999977
Epoch: 862, Loss: 0.17999999999999994
Epoch: 863, Loss: 0.17999999999999977
Epoch: 864, Loss: 0.17999999999999994
Epoch: 865, Loss: 0.17999999999999977
Epoch: 866, Loss: 0.17999999999999977
Epoch: 867, Loss: 0.17999999999999994
Epoch: 868, Loss: 0.17999999999999977
Epoch: 869, Loss: 0.17999999999999994
Epoch: 870, Loss: 0.17999999999999977
Epoch: 871, Loss: 0.17999999999999977
Epoch: 872, Loss: 0.17999999999999994
Epoch: 873, Loss: 0.17999999999999977
Epoch: 874, Loss: 0.17999999999999994
Epoch: 875, Loss: 0.17999999999999977
Epoch: 876, Loss: 0.17999999999999977
Epoch: 877, Loss: 0.17999999999999994
Epoch: 878, Loss: 0.17999999999999977
Epoch: 879, Loss: 0.17999999999999994
Epoch: 880, Loss: 0.17999999999999977
Epoch: 881, Loss: 0.17999999999999977
Epoch: 882, Loss: 0.17999999999999994
Epoch: 883, Loss: 0.17999999999999977
Epoch: 884, Loss: 0.17999999999999994
Epoch: 885, Loss: 0.17999999999999977
Epoch: 886, Loss: 0.17999999999999994
Epoch: 887, Loss: 0.17999999999999977
Epoch: 888, Loss: 0.17999999999999977
Epoch: 889, Loss: 0.17999999999999994
Epoch: 890, Loss: 0.17999999999999977
Epoch: 891, Loss: 0.17999999999999994
Epoch: 892, Loss: 0.17999999999999977
Epoch: 893, Loss: 0.17999999999999977
Epoch: 894, Loss: 0.17999999999999994
Epoch: 895, Loss: 0.17999999999999977
Epoch: 896, Loss: 0.17999999999999994
Epoch: 897, Loss: 0.17999999999999977
Epoch: 898, Loss: 0.17999999999999994
Epoch: 899, Loss: 0.17999999999999977
Epoch: 900, Loss: 0.17999999999999977
Epoch: 901, Loss: 0.17999999999999994
Epoch: 902, Loss: 0.17999999999999977
Epoch: 903, Loss: 0.17999999999999994
Epoch: 904, Loss: 0.17999999999999977
Epoch: 905, Loss: 0.17999999999999977
Epoch: 906, Loss: 0.17999999999999994
Epoch: 907, Loss: 0.17999999999999977
Epoch: 908, Loss: 0.17999999999999994
Epoch: 909, Loss: 0.17999999999999977
Epoch: 910, Loss: 0.17999999999999977
Epoch: 911, Loss: 0.17999999999999994
Epoch: 912, Loss: 0.17999999999999977
Epoch: 913, Loss: 0.17999999999999994
Epoch: 914, Loss: 0.17999999999999977
Epoch: 915, Loss: 0.17999999999999977
Epoch: 916, Loss: 0.17999999999999994
Epoch: 917, Loss: 0.17999999999999977
Epoch: 918, Loss: 0.17999999999999994
Epoch: 919, Loss: 0.17999999999999977
Epoch: 920, Loss: 0.17999999999999994
Epoch: 921, Loss: 0.17999999999999977
Epoch: 922, Loss: 0.17999999999999977
Epoch: 923, Loss: 0.17999999999999994
Epoch: 924, Loss: 0.17999999999999977
Epoch: 925, Loss: 0.17999999999999994
Epoch: 926, Loss: 0.17999999999999977
Epoch: 927, Loss: 0.17999999999999977
Epoch: 928, Loss: 0.17999999999999994
Epoch: 929, Loss: 0.17999999999999977
Epoch: 930, Loss: 0.17999999999999994
Epoch: 931, Loss: 0.17999999999999977
Epoch: 932, Loss: 0.17999999999999994
Epoch: 933, Loss: 0.17999999999999977
Epoch: 934, Loss: 0.17999999999999977
Epoch: 935, Loss: 0.17999999999999994
Epoch: 936, Loss: 0.17999999999999977
Epoch: 937, Loss: 0.17999999999999977
Epoch: 938, Loss: 0.17999999999999994
Epoch: 939, Loss: 0.17999999999999977
Epoch: 940, Loss: 0.17999999999999994
Epoch: 941, Loss: 0.17999999999999977
Epoch: 942, Loss: 0.17999999999999977
Epoch: 943, Loss: 0.17999999999999994
Epoch: 944, Loss: 0.17999999999999977
Epoch: 945, Loss: 0.17999999999999994
Epoch: 946, Loss: 0.17999999999999977
Epoch: 947, Loss: 0.17999999999999977
Epoch: 948, Loss: 0.17999999999999994
Epoch: 949, Loss: 0.17999999999999977
Epoch: 950, Loss: 0.17999999999999994
Epoch: 951, Loss: 0.17999999999999977
Epoch: 952, Loss: 0.17999999999999977
Epoch: 953, Loss: 0.17999999999999994
Epoch: 954, Loss: 0.17999999999999977
Epoch: 955, Loss: 0.17999999999999994
Epoch: 956, Loss: 0.17999999999999977
Epoch: 957, Loss: 0.17999999999999977
Epoch: 958, Loss: 0.17999999999999994
Epoch: 959, Loss: 0.17999999999999977
Epoch: 960, Loss: 0.17999999999999994
Epoch: 961, Loss: 0.17999999999999977
Epoch: 962, Loss: 0.17999999999999977
Epoch: 963, Loss: 0.17999999999999994
Epoch: 964, Loss: 0.17999999999999977
Epoch: 965, Loss: 0.17999999999999994
Epoch: 966, Loss: 0.17999999999999977
Epoch: 967, Loss: 0.17999999999999977
Epoch: 968, Loss: 0.17999999999999994
Epoch: 969, Loss: 0.17999999999999977
Epoch: 970, Loss: 0.17999999999999994
Epoch: 971, Loss: 0.17999999999999977
Epoch: 972, Loss: 0.17999999999999977
Epoch: 973, Loss: 0.17999999999999994
Epoch: 974, Loss: 0.17999999999999977
Epoch: 975, Loss: 0.17999999999999994
Epoch: 976, Loss: 0.17999999999999977
Epoch: 977, Loss: 0.17999999999999977
Epoch: 978, Loss: 0.17999999999999994
Epoch: 979, Loss: 0.17999999999999977
Epoch: 980, Loss: 0.17999999999999994
Epoch: 981, Loss: 0.17999999999999977
Epoch: 982, Loss: 0.17999999999999977
Epoch: 983, Loss: 0.17999999999999994
Epoch: 984, Loss: 0.17999999999999977
Epoch: 985, Loss: 0.17999999999999994
Epoch: 986, Loss: 0.17999999999999977
Epoch: 987, Loss: 0.17999999999999977
Epoch: 988, Loss: 0.17999999999999994
Epoch: 989, Loss: 0.17999999999999977
Epoch: 990, Loss: 0.17999999999999994
Epoch: 991, Loss: 0.17999999999999977
Epoch: 992, Loss: 0.17999999999999994
Epoch: 993, Loss: 0.17999999999999977
Epoch: 994, Loss: 0.17999999999999977
Epoch: 995, Loss: 0.17999999999999994
Epoch: 996, Loss: 0.17999999999999977
Epoch: 997, Loss: 0.17999999999999994
Epoch: 998, Loss: 0.17999999999999977
Epoch: 999, Loss: 0.17999999999999994
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library %  python3 scratch/train_loop.py
Epoch: 0, Loss: 12.443735497744838
Epoch: 1, Loss: 13.48188790041818
Epoch: 2, Loss: 11.90561369972375
Epoch: 3, Loss: 6.927243624552367
Epoch: 4, Loss: 3.229436386021298
Epoch: 5, Loss: 1.2326501675329542
Epoch: 6, Loss: 0.534480437695644
Epoch: 7, Loss: 0.29582706877302356
Epoch: 8, Loss: 0.21988823354706327
Epoch: 9, Loss: 0.19447867650196865
Epoch: 10, Loss: 0.18571219367571315
Epoch: 11, Loss: 0.18245629279027128
Epoch: 12, Loss: 0.1811486914224178
Epoch: 13, Loss: 0.18057444778906523
Epoch: 14, Loss: 0.18030146067350766
Epoch: 15, Loss: 0.1801632120288123
Epoch: 16, Loss: 0.18009006508388803
Epoch: 17, Loss: 0.18005025892876733
Epoch: 18, Loss: 0.18002822633986107
Epoch: 19, Loss: 0.18001590991890234
Epoch: 20, Loss: 0.18000898591670675
Epoch: 21, Loss: 0.18000508096903176
Epoch: 22, Loss: 0.18000287476082524
Epoch: 23, Loss: 0.18000162706794076
Epoch: 24, Loss: 0.18000092106611906
Epoch: 25, Loss: 0.1800005214583564
Epoch: 26, Loss: 0.18000029523759414
Epoch: 27, Loss: 0.1800001671612797
Epoch: 28, Loss: 0.18000009464675176
Epoch: 29, Loss: 0.1800000535893561
Epoch: 30, Loss: 0.18000003034258086
Epoch: 31, Loss: 0.1800000171801449
Epoch: 32, Loss: 0.18000000972749633
Epoch: 33, Loss: 0.18000000550776152
Epoch: 34, Loss: 0.18000000311852313
Epoch: 35, Loss: 0.1800000017657228
Epoch: 36, Loss: 0.18000000099975993
Epoch: 37, Loss: 0.18000000056606857
Epoch: 38, Loss: 0.18000000032050975
Epoch: 39, Loss: 0.1800000001814746
Epoch: 40, Loss: 0.18000000010275197
Epoch: 41, Loss: 0.1800000000581779
Epoch: 42, Loss: 0.18000000003294087
Epoch: 43, Loss: 0.18000000001865074
Epoch: 44, Loss: 0.18000000001056043
Epoch: 45, Loss: 0.18000000000598074
Epoch: 46, Loss: 0.18000000000338534
Epoch: 47, Loss: 0.18000000000191635
Epoch: 48, Loss: 0.1800000000010852
Epoch: 49, Loss: 0.18000000000061464
Epoch: 50, Loss: 0.18000000000034821
Epoch: 51, Loss: 0.1800000000001968
Epoch: 52, Loss: 0.18000000000011207
Epoch: 53, Loss: 0.18000000000006305
Epoch: 54, Loss: 0.1800000000000358
Epoch: 55, Loss: 0.18000000000001978
Epoch: 56, Loss: 0.1800000000000112
Epoch: 57, Loss: 0.1800000000000067
Epoch: 58, Loss: 0.1800000000000036
Epoch: 59, Loss: 0.18000000000000238
Epoch: 60, Loss: 0.1800000000000014
Epoch: 61, Loss: 0.1800000000000015
Epoch: 62, Loss: 0.1800000000000006
Epoch: 63, Loss: 0.17999999999999977
Epoch: 64, Loss: 0.18000000000000066
Epoch: 65, Loss: 0.18000000000000013
Epoch: 66, Loss: 0.1799999999999996
Epoch: 67, Loss: 0.18000000000000005
Epoch: 68, Loss: 0.1800000000000001
Epoch: 69, Loss: 0.17999999999999977
Epoch: 70, Loss: 0.17999999999999994
Epoch: 71, Loss: 0.1799999999999996
Epoch: 72, Loss: 0.1800000000000002
Epoch: 73, Loss: 0.18000000000000071
Epoch: 74, Loss: 0.17999999999999977
Epoch: 75, Loss: 0.1800000000000003
Epoch: 76, Loss: 0.18000000000000022
Epoch: 77, Loss: 0.18000000000000033
Epoch: 78, Loss: 0.17999999999999963
Epoch: 79, Loss: 0.18000000000000038
Epoch: 80, Loss: 0.18000000000000016
Epoch: 81, Loss: 0.17999999999999916
Epoch: 82, Loss: 0.1799999999999999
Epoch: 83, Loss: 0.1800000000000001
Epoch: 84, Loss: 0.17999999999999994
Epoch: 85, Loss: 0.18000000000000005
Epoch: 86, Loss: 0.17999999999999988
Epoch: 87, Loss: 0.18
Epoch: 88, Loss: 0.18
Epoch: 89, Loss: 0.17999999999999983
Epoch: 90, Loss: 0.1799999999999997
Epoch: 91, Loss: 0.17999999999999955
Epoch: 92, Loss: 0.1799999999999995
Epoch: 93, Loss: 0.1800000000000006
Epoch: 94, Loss: 0.17999999999999983
Epoch: 95, Loss: 0.17999999999999994
Epoch: 96, Loss: 0.1800000000000005
Epoch: 97, Loss: 0.18000000000000005
Epoch: 98, Loss: 0.17999999999999988
Epoch: 99, Loss: 0.1799999999999996
Epoch: 100, Loss: 0.18000000000000038
Epoch: 101, Loss: 0.1800000000000001
Epoch: 102, Loss: 0.17999999999999972
Epoch: 103, Loss: 0.1800000000000004
Epoch: 104, Loss: 0.17999999999999983
Epoch: 105, Loss: 0.18000000000000008
Epoch: 106, Loss: 0.17999999999999988
Epoch: 107, Loss: 0.18000000000000058
Epoch: 108, Loss: 0.18000000000000022
Epoch: 109, Loss: 0.1800000000000004
Epoch: 110, Loss: 0.17999999999999983
Epoch: 111, Loss: 0.17999999999999994
Epoch: 112, Loss: 0.1800000000000006
Epoch: 113, Loss: 0.1800000000000008
Epoch: 114, Loss: 0.1800000000000004
Epoch: 115, Loss: 0.18000000000000083
Epoch: 116, Loss: 0.1800000000000007
Epoch: 117, Loss: 0.18000000000000066
Epoch: 118, Loss: 0.180000000000001
Epoch: 119, Loss: 0.18000000000000083
Epoch: 120, Loss: 0.18000000000000038
Epoch: 121, Loss: 0.18000000000000063
Epoch: 122, Loss: 0.18000000000000063
Epoch: 123, Loss: 0.18000000000000058
Epoch: 124, Loss: 0.18000000000000027
Epoch: 125, Loss: 0.18000000000000033
Epoch: 126, Loss: 0.18000000000000016
Epoch: 127, Loss: 0.18000000000000027
Epoch: 128, Loss: 0.1800000000000003
Epoch: 129, Loss: 0.1799999999999999
Epoch: 130, Loss: 0.17999999999999994
Epoch: 131, Loss: 0.17999999999999988
Epoch: 132, Loss: 0.18000000000000013
Epoch: 133, Loss: 0.18000000000000055
Epoch: 134, Loss: 0.18000000000000052
Epoch: 135, Loss: 0.18000000000000071
Epoch: 136, Loss: 0.18000000000000055
Epoch: 137, Loss: 0.18000000000000055
Epoch: 138, Loss: 0.1800000000000007
Epoch: 139, Loss: 0.18000000000000077
Epoch: 140, Loss: 0.18000000000000077
Epoch: 141, Loss: 0.1800000000000008
Epoch: 142, Loss: 0.18000000000000071
Epoch: 143, Loss: 0.18000000000000071
Epoch: 144, Loss: 0.18000000000000083
Epoch: 145, Loss: 0.18000000000000083
Epoch: 146, Loss: 0.18000000000000083
Epoch: 147, Loss: 0.18000000000000083
Epoch: 148, Loss: 0.18000000000000094
Epoch: 149, Loss: 0.18000000000000088
Epoch: 150, Loss: 0.18000000000000071
Epoch: 151, Loss: 0.18000000000000094
Epoch: 152, Loss: 0.18000000000000088
Epoch: 153, Loss: 0.18000000000000094
Epoch: 154, Loss: 0.18000000000000094
Epoch: 155, Loss: 0.18000000000000094
Epoch: 156, Loss: 0.18000000000000094
Epoch: 157, Loss: 0.18000000000000094
Epoch: 158, Loss: 0.18000000000000094
Epoch: 159, Loss: 0.18000000000000094
Epoch: 160, Loss: 0.18000000000000094
Epoch: 161, Loss: 0.18000000000000094
Epoch: 162, Loss: 0.18000000000000094
Epoch: 163, Loss: 0.18000000000000094
Epoch: 164, Loss: 0.18000000000000094
Epoch: 165, Loss: 0.18000000000000094
Epoch: 166, Loss: 0.18000000000000094
Epoch: 167, Loss: 0.18000000000000094
Epoch: 168, Loss: 0.18000000000000083
Epoch: 169, Loss: 0.1800000000000007
Epoch: 170, Loss: 0.18000000000000085
Epoch: 171, Loss: 0.18000000000000085
Epoch: 172, Loss: 0.18000000000000083
Epoch: 173, Loss: 0.18000000000000083
Epoch: 174, Loss: 0.18000000000000077
Epoch: 175, Loss: 0.18000000000000088
Epoch: 176, Loss: 0.18000000000000094
Epoch: 177, Loss: 0.1800000000000006
Epoch: 178, Loss: 0.1800000000000006
Epoch: 179, Loss: 0.18000000000000027
Epoch: 180, Loss: 0.18000000000000033
Epoch: 181, Loss: 0.1800000000000003
Epoch: 182, Loss: 0.1800000000000003
Epoch: 183, Loss: 0.18000000000000027
Epoch: 184, Loss: 0.18000000000000033
Epoch: 185, Loss: 0.18000000000000033
Epoch: 186, Loss: 0.18000000000000022
Epoch: 187, Loss: 0.18000000000000022
Epoch: 188, Loss: 0.18000000000000016
Epoch: 189, Loss: 0.18000000000000027
Epoch: 190, Loss: 0.18000000000000027
Epoch: 191, Loss: 0.18000000000000022
Epoch: 192, Loss: 0.1800000000000002
Epoch: 193, Loss: 0.18000000000000027
Epoch: 194, Loss: 0.18000000000000038
Epoch: 195, Loss: 0.18000000000000035
Epoch: 196, Loss: 0.18000000000000035
Epoch: 197, Loss: 0.18000000000000027
Epoch: 198, Loss: 0.18000000000000024
Epoch: 199, Loss: 0.18000000000000024
Epoch: 200, Loss: 0.18000000000000008
Epoch: 201, Loss: 0.18000000000000013
Epoch: 202, Loss: 0.18000000000000013
Epoch: 203, Loss: 0.18000000000000016
Epoch: 204, Loss: 0.1800000000000001
Epoch: 205, Loss: 0.1800000000000001
Epoch: 206, Loss: 0.1800000000000001
Epoch: 207, Loss: 0.1800000000000001
Epoch: 208, Loss: 0.1800000000000001
Epoch: 209, Loss: 0.18000000000000027
Epoch: 210, Loss: 0.18000000000000022
Epoch: 211, Loss: 0.18000000000000022
Epoch: 212, Loss: 0.18000000000000022
Epoch: 213, Loss: 0.18000000000000022
Epoch: 214, Loss: 0.18000000000000022
Epoch: 215, Loss: 0.18000000000000066
Epoch: 216, Loss: 0.18000000000000066
Epoch: 217, Loss: 0.18000000000000066
Epoch: 218, Loss: 0.18000000000000066
Epoch: 219, Loss: 0.18000000000000066
Epoch: 220, Loss: 0.18000000000000055
Epoch: 221, Loss: 0.18000000000000055
Epoch: 222, Loss: 0.18000000000000055
Epoch: 223, Loss: 0.18000000000000055
Epoch: 224, Loss: 0.18000000000000055
Epoch: 225, Loss: 0.18000000000000055
Epoch: 226, Loss: 0.18000000000000052
Epoch: 227, Loss: 0.18000000000000052
Epoch: 228, Loss: 0.18000000000000055
Epoch: 229, Loss: 0.1800000000000005
Epoch: 230, Loss: 0.1800000000000005
Epoch: 231, Loss: 0.1800000000000005
Epoch: 232, Loss: 0.1800000000000005
Epoch: 233, Loss: 0.1800000000000006
Epoch: 234, Loss: 0.18000000000000063
Epoch: 235, Loss: 0.18000000000000055
Epoch: 236, Loss: 0.1800000000000005
Epoch: 237, Loss: 0.1800000000000005
Epoch: 238, Loss: 0.1800000000000006
Epoch: 239, Loss: 0.18000000000000063
Epoch: 240, Loss: 0.18000000000000063
Epoch: 241, Loss: 0.18000000000000063
Epoch: 242, Loss: 0.18000000000000063
Epoch: 243, Loss: 0.18000000000000063
Epoch: 244, Loss: 0.18000000000000063
Epoch: 245, Loss: 0.18000000000000063
Epoch: 246, Loss: 0.18000000000000063
Epoch: 247, Loss: 0.18000000000000063
Epoch: 248, Loss: 0.18000000000000063
Epoch: 249, Loss: 0.18000000000000063
Epoch: 250, Loss: 0.18000000000000063
Epoch: 251, Loss: 0.18000000000000063
Epoch: 252, Loss: 0.18000000000000063
Epoch: 253, Loss: 0.18000000000000063
Epoch: 254, Loss: 0.18000000000000063
Epoch: 255, Loss: 0.18000000000000063
Epoch: 256, Loss: 0.18000000000000063
Epoch: 257, Loss: 0.18000000000000063
Epoch: 258, Loss: 0.18000000000000063
Epoch: 259, Loss: 0.18000000000000063
Epoch: 260, Loss: 0.18000000000000063
Epoch: 261, Loss: 0.18000000000000063
Epoch: 262, Loss: 0.18000000000000063
Epoch: 263, Loss: 0.1800000000000008
Epoch: 264, Loss: 0.1800000000000008
Epoch: 265, Loss: 0.1800000000000008
Epoch: 266, Loss: 0.18000000000000063
Epoch: 267, Loss: 0.1800000000000008
Epoch: 268, Loss: 0.18000000000000063
Epoch: 269, Loss: 0.18000000000000077
Epoch: 270, Loss: 0.18000000000000063
Epoch: 271, Loss: 0.18000000000000066
Epoch: 272, Loss: 0.18000000000000063
Epoch: 273, Loss: 0.18000000000000063
Epoch: 274, Loss: 0.18000000000000077
Epoch: 275, Loss: 0.18000000000000063
Epoch: 276, Loss: 0.18000000000000066
Epoch: 277, Loss: 0.18000000000000063
Epoch: 278, Loss: 0.18000000000000063
Epoch: 279, Loss: 0.18000000000000077
Epoch: 280, Loss: 0.18000000000000063
Epoch: 281, Loss: 0.18000000000000066
Epoch: 282, Loss: 0.18000000000000063
Epoch: 283, Loss: 0.18000000000000063
Epoch: 284, Loss: 0.18000000000000077
Epoch: 285, Loss: 0.18000000000000063
Epoch: 286, Loss: 0.18000000000000066
Epoch: 287, Loss: 0.18000000000000063
Epoch: 288, Loss: 0.18000000000000063
Epoch: 289, Loss: 0.18000000000000077
Epoch: 290, Loss: 0.18000000000000063
Epoch: 291, Loss: 0.18000000000000066
Epoch: 292, Loss: 0.18000000000000063
Epoch: 293, Loss: 0.18000000000000063
Epoch: 294, Loss: 0.18000000000000077
Epoch: 295, Loss: 0.18000000000000063
Epoch: 296, Loss: 0.18000000000000066
Epoch: 297, Loss: 0.18000000000000063
Epoch: 298, Loss: 0.18000000000000063
Epoch: 299, Loss: 0.18000000000000077
Epoch: 300, Loss: 0.18000000000000063
Epoch: 301, Loss: 0.18000000000000066
Epoch: 302, Loss: 0.18000000000000063
Epoch: 303, Loss: 0.18000000000000063
Epoch: 304, Loss: 0.18000000000000077
Epoch: 305, Loss: 0.18000000000000063
Epoch: 306, Loss: 0.18000000000000066
Epoch: 307, Loss: 0.18000000000000063
Epoch: 308, Loss: 0.18000000000000063
Epoch: 309, Loss: 0.18000000000000077
Epoch: 310, Loss: 0.18000000000000063
Epoch: 311, Loss: 0.18000000000000066
Epoch: 312, Loss: 0.18000000000000063
Epoch: 313, Loss: 0.18000000000000063
Epoch: 314, Loss: 0.18000000000000063
Epoch: 315, Loss: 0.18000000000000077
Epoch: 316, Loss: 0.18000000000000063
Epoch: 317, Loss: 0.18000000000000066
Epoch: 318, Loss: 0.18000000000000063
Epoch: 319, Loss: 0.18000000000000063
Epoch: 320, Loss: 0.18000000000000077
Epoch: 321, Loss: 0.18000000000000063
Epoch: 322, Loss: 0.18000000000000066
Epoch: 323, Loss: 0.18000000000000063
Epoch: 324, Loss: 0.18000000000000063
Epoch: 325, Loss: 0.18000000000000077
Epoch: 326, Loss: 0.18000000000000063
Epoch: 327, Loss: 0.18000000000000066
Epoch: 328, Loss: 0.18000000000000063
Epoch: 329, Loss: 0.18000000000000063
Epoch: 330, Loss: 0.18000000000000077
Epoch: 331, Loss: 0.18000000000000063
Epoch: 332, Loss: 0.18000000000000066
Epoch: 333, Loss: 0.18000000000000063
Epoch: 334, Loss: 0.18000000000000063
Epoch: 335, Loss: 0.18000000000000077
Epoch: 336, Loss: 0.18000000000000063
Epoch: 337, Loss: 0.18000000000000066
Epoch: 338, Loss: 0.18000000000000063
Epoch: 339, Loss: 0.18000000000000063
Epoch: 340, Loss: 0.18000000000000077
Epoch: 341, Loss: 0.18000000000000063
Epoch: 342, Loss: 0.18000000000000066
Epoch: 343, Loss: 0.18000000000000063
Epoch: 344, Loss: 0.18000000000000077
Epoch: 345, Loss: 0.18000000000000063
Epoch: 346, Loss: 0.18000000000000066
Epoch: 347, Loss: 0.18000000000000063
Epoch: 348, Loss: 0.18000000000000063
Epoch: 349, Loss: 0.18000000000000077
Epoch: 350, Loss: 0.18000000000000063
Epoch: 351, Loss: 0.18000000000000066
Epoch: 352, Loss: 0.18000000000000063
Epoch: 353, Loss: 0.18000000000000063
Epoch: 354, Loss: 0.18000000000000077
Epoch: 355, Loss: 0.18000000000000063
Epoch: 356, Loss: 0.18000000000000066
Epoch: 357, Loss: 0.18000000000000063
Epoch: 358, Loss: 0.18000000000000063
Epoch: 359, Loss: 0.18000000000000077
Epoch: 360, Loss: 0.18000000000000063
Epoch: 361, Loss: 0.18000000000000066
Epoch: 362, Loss: 0.18000000000000063
Epoch: 363, Loss: 0.18000000000000077
Epoch: 364, Loss: 0.18000000000000063
Epoch: 365, Loss: 0.18000000000000066
Epoch: 366, Loss: 0.18000000000000063
Epoch: 367, Loss: 0.18000000000000063
Epoch: 368, Loss: 0.18000000000000077
Epoch: 369, Loss: 0.18000000000000063
Epoch: 370, Loss: 0.18000000000000066
Epoch: 371, Loss: 0.18000000000000063
Epoch: 372, Loss: 0.18000000000000063
Epoch: 373, Loss: 0.18000000000000077
Epoch: 374, Loss: 0.18000000000000063
Epoch: 375, Loss: 0.18000000000000066
Epoch: 376, Loss: 0.18000000000000063
Epoch: 377, Loss: 0.18000000000000063
Epoch: 378, Loss: 0.18000000000000077
Epoch: 379, Loss: 0.18000000000000063
Epoch: 380, Loss: 0.18000000000000066
Epoch: 381, Loss: 0.18000000000000063
Epoch: 382, Loss: 0.18000000000000063
Epoch: 383, Loss: 0.18000000000000077
Epoch: 384, Loss: 0.18000000000000063
Epoch: 385, Loss: 0.18000000000000066
Epoch: 386, Loss: 0.18000000000000063
Epoch: 387, Loss: 0.18000000000000063
Epoch: 388, Loss: 0.18000000000000077
Epoch: 389, Loss: 0.18000000000000063
Epoch: 390, Loss: 0.18000000000000066
Epoch: 391, Loss: 0.18000000000000063
Epoch: 392, Loss: 0.18000000000000063
Epoch: 393, Loss: 0.18000000000000077
Epoch: 394, Loss: 0.18000000000000063
Epoch: 395, Loss: 0.18000000000000066
Epoch: 396, Loss: 0.18000000000000063
Epoch: 397, Loss: 0.18000000000000077
Epoch: 398, Loss: 0.18000000000000063
Epoch: 399, Loss: 0.18000000000000066
Epoch: 400, Loss: 0.18000000000000063
Epoch: 401, Loss: 0.18000000000000063
Epoch: 402, Loss: 0.18000000000000077
Epoch: 403, Loss: 0.18000000000000063
Epoch: 404, Loss: 0.18000000000000066
Epoch: 405, Loss: 0.18000000000000063
Epoch: 406, Loss: 0.18000000000000063
Epoch: 407, Loss: 0.18000000000000077
Epoch: 408, Loss: 0.18000000000000063
Epoch: 409, Loss: 0.18000000000000066
Epoch: 410, Loss: 0.18000000000000063
Epoch: 411, Loss: 0.18000000000000063
Epoch: 412, Loss: 0.18000000000000077
Epoch: 413, Loss: 0.18000000000000063
Epoch: 414, Loss: 0.18000000000000066
Epoch: 415, Loss: 0.18000000000000063
Epoch: 416, Loss: 0.18000000000000063
Epoch: 417, Loss: 0.18000000000000077
Epoch: 418, Loss: 0.18000000000000063
Epoch: 419, Loss: 0.18000000000000066
Epoch: 420, Loss: 0.18000000000000063
Epoch: 421, Loss: 0.18000000000000063
Epoch: 422, Loss: 0.18000000000000077
Epoch: 423, Loss: 0.18000000000000063
Epoch: 424, Loss: 0.18000000000000066
Epoch: 425, Loss: 0.18000000000000063
Epoch: 426, Loss: 0.18000000000000063
Epoch: 427, Loss: 0.18000000000000077
Epoch: 428, Loss: 0.18000000000000063
Epoch: 429, Loss: 0.18000000000000066
Epoch: 430, Loss: 0.18000000000000063
Epoch: 431, Loss: 0.18000000000000063
Epoch: 432, Loss: 0.18000000000000077
Epoch: 433, Loss: 0.18000000000000063
Epoch: 434, Loss: 0.18000000000000066
Epoch: 435, Loss: 0.18000000000000063
Epoch: 436, Loss: 0.18000000000000063
Epoch: 437, Loss: 0.18000000000000077
Epoch: 438, Loss: 0.18000000000000063
Epoch: 439, Loss: 0.18000000000000066
Epoch: 440, Loss: 0.18000000000000063
Epoch: 441, Loss: 0.18000000000000063
Epoch: 442, Loss: 0.18000000000000077
Epoch: 443, Loss: 0.18000000000000063
Epoch: 444, Loss: 0.18000000000000066
Epoch: 445, Loss: 0.18000000000000063
Epoch: 446, Loss: 0.18000000000000063
Epoch: 447, Loss: 0.18000000000000077
Epoch: 448, Loss: 0.18000000000000063
Epoch: 449, Loss: 0.18000000000000066
Epoch: 450, Loss: 0.18000000000000063
Epoch: 451, Loss: 0.18000000000000063
Epoch: 452, Loss: 0.18000000000000077
Epoch: 453, Loss: 0.18000000000000063
Epoch: 454, Loss: 0.18000000000000066
Epoch: 455, Loss: 0.18000000000000063
Epoch: 456, Loss: 0.18000000000000063
Epoch: 457, Loss: 0.18000000000000077
Epoch: 458, Loss: 0.18000000000000063
Epoch: 459, Loss: 0.18000000000000066
Epoch: 460, Loss: 0.18000000000000063
Epoch: 461, Loss: 0.18000000000000063
Epoch: 462, Loss: 0.18000000000000077
Epoch: 463, Loss: 0.18000000000000063
Epoch: 464, Loss: 0.18000000000000066
Epoch: 465, Loss: 0.18000000000000063
Epoch: 466, Loss: 0.18000000000000063
Epoch: 467, Loss: 0.18000000000000077
Epoch: 468, Loss: 0.18000000000000063
Epoch: 469, Loss: 0.18000000000000066
Epoch: 470, Loss: 0.18000000000000063
Epoch: 471, Loss: 0.18000000000000063
Epoch: 472, Loss: 0.18000000000000077
Epoch: 473, Loss: 0.18000000000000063
Epoch: 474, Loss: 0.18000000000000066
Epoch: 475, Loss: 0.18000000000000063
Epoch: 476, Loss: 0.18000000000000077
Epoch: 477, Loss: 0.18000000000000063
Epoch: 478, Loss: 0.18000000000000066
Epoch: 479, Loss: 0.18000000000000063
Epoch: 480, Loss: 0.18000000000000063
Epoch: 481, Loss: 0.18000000000000077
Epoch: 482, Loss: 0.18000000000000063
Epoch: 483, Loss: 0.18000000000000066
Epoch: 484, Loss: 0.18000000000000063
Epoch: 485, Loss: 0.18000000000000063
Epoch: 486, Loss: 0.18000000000000077
Epoch: 487, Loss: 0.18000000000000063
Epoch: 488, Loss: 0.18000000000000066
Epoch: 489, Loss: 0.18000000000000063
Epoch: 490, Loss: 0.18000000000000063
Epoch: 491, Loss: 0.18000000000000077
Epoch: 492, Loss: 0.18000000000000063
Epoch: 493, Loss: 0.18000000000000066
Epoch: 494, Loss: 0.18000000000000063
Epoch: 495, Loss: 0.18000000000000063
Epoch: 496, Loss: 0.18000000000000077
Epoch: 497, Loss: 0.18000000000000063
Epoch: 498, Loss: 0.18000000000000066
Epoch: 499, Loss: 0.18000000000000063
Epoch: 500, Loss: 0.18000000000000063
Epoch: 501, Loss: 0.18000000000000077
Epoch: 502, Loss: 0.18000000000000063
Epoch: 503, Loss: 0.18000000000000066
Epoch: 504, Loss: 0.18000000000000063
Epoch: 505, Loss: 0.18000000000000063
Epoch: 506, Loss: 0.18000000000000077
Epoch: 507, Loss: 0.18000000000000063
Epoch: 508, Loss: 0.18000000000000066
Epoch: 509, Loss: 0.18000000000000063
Epoch: 510, Loss: 0.18000000000000063
Epoch: 511, Loss: 0.18000000000000077
Epoch: 512, Loss: 0.18000000000000063
Epoch: 513, Loss: 0.18000000000000066
Epoch: 514, Loss: 0.18000000000000063
Epoch: 515, Loss: 0.18000000000000063
Epoch: 516, Loss: 0.18000000000000063
Epoch: 517, Loss: 0.18000000000000077
Epoch: 518, Loss: 0.18000000000000063
Epoch: 519, Loss: 0.18000000000000066
Epoch: 520, Loss: 0.18000000000000063
Epoch: 521, Loss: 0.18000000000000063
Epoch: 522, Loss: 0.18000000000000077
Epoch: 523, Loss: 0.18000000000000063
Epoch: 524, Loss: 0.18000000000000066
Epoch: 525, Loss: 0.18000000000000063
Epoch: 526, Loss: 0.18000000000000063
Epoch: 527, Loss: 0.18000000000000077
Epoch: 528, Loss: 0.18000000000000063
Epoch: 529, Loss: 0.18000000000000066
Epoch: 530, Loss: 0.18000000000000063
Epoch: 531, Loss: 0.18000000000000063
Epoch: 532, Loss: 0.18000000000000077
Epoch: 533, Loss: 0.18000000000000063
Epoch: 534, Loss: 0.18000000000000066
Epoch: 535, Loss: 0.18000000000000063
Epoch: 536, Loss: 0.18000000000000063
Epoch: 537, Loss: 0.18000000000000077
Epoch: 538, Loss: 0.18000000000000063
Epoch: 539, Loss: 0.18000000000000066
Epoch: 540, Loss: 0.18000000000000063
Epoch: 541, Loss: 0.18000000000000063
Epoch: 542, Loss: 0.18000000000000077
Epoch: 543, Loss: 0.18000000000000063
Epoch: 544, Loss: 0.18000000000000066
Epoch: 545, Loss: 0.18000000000000063
Epoch: 546, Loss: 0.18000000000000063
Epoch: 547, Loss: 0.18000000000000077
Epoch: 548, Loss: 0.18000000000000033
Epoch: 549, Loss: 0.18000000000000047
Epoch: 550, Loss: 0.18000000000000085
Epoch: 551, Loss: 0.18000000000000055
Epoch: 552, Loss: 0.18000000000000088
Epoch: 553, Loss: 0.18000000000000058
Epoch: 554, Loss: 0.18000000000000088
Epoch: 555, Loss: 0.18000000000000058
Epoch: 556, Loss: 0.18000000000000071
Epoch: 557, Loss: 0.18000000000000071
Epoch: 558, Loss: 0.18000000000000071
Epoch: 559, Loss: 0.18000000000000071
Epoch: 560, Loss: 0.18000000000000071
Epoch: 561, Loss: 0.18000000000000071
Epoch: 562, Loss: 0.18000000000000071
Epoch: 563, Loss: 0.18000000000000071
Epoch: 564, Loss: 0.18000000000000071
Epoch: 565, Loss: 0.18000000000000071
Epoch: 566, Loss: 0.18000000000000071
Epoch: 567, Loss: 0.18000000000000071
Epoch: 568, Loss: 0.18000000000000071
Epoch: 569, Loss: 0.18000000000000071
Epoch: 570, Loss: 0.18000000000000071
Epoch: 571, Loss: 0.18000000000000071
Epoch: 572, Loss: 0.18000000000000071
Epoch: 573, Loss: 0.18000000000000071
Epoch: 574, Loss: 0.18000000000000071
Epoch: 575, Loss: 0.18000000000000071
Epoch: 576, Loss: 0.18000000000000071
Epoch: 577, Loss: 0.18000000000000071
Epoch: 578, Loss: 0.18000000000000071
Epoch: 579, Loss: 0.18000000000000071
Epoch: 580, Loss: 0.18000000000000071
Epoch: 581, Loss: 0.18000000000000071
Epoch: 582, Loss: 0.18000000000000071
Epoch: 583, Loss: 0.18000000000000071
Epoch: 584, Loss: 0.18000000000000071
Epoch: 585, Loss: 0.18000000000000071
Epoch: 586, Loss: 0.18000000000000071
Epoch: 587, Loss: 0.18000000000000071
Epoch: 588, Loss: 0.18000000000000071
Epoch: 589, Loss: 0.18000000000000071
Epoch: 590, Loss: 0.18000000000000071
Epoch: 591, Loss: 0.18000000000000071
Epoch: 592, Loss: 0.18000000000000071
Epoch: 593, Loss: 0.18000000000000071
Epoch: 594, Loss: 0.18000000000000071
Epoch: 595, Loss: 0.18000000000000071
Epoch: 596, Loss: 0.18000000000000071
Epoch: 597, Loss: 0.18000000000000071
Epoch: 598, Loss: 0.18000000000000071
Epoch: 599, Loss: 0.18000000000000071
Epoch: 600, Loss: 0.18000000000000071
Epoch: 601, Loss: 0.18000000000000071
Epoch: 602, Loss: 0.18000000000000071
Epoch: 603, Loss: 0.18000000000000071
Epoch: 604, Loss: 0.18000000000000071
Epoch: 605, Loss: 0.18000000000000071
Epoch: 606, Loss: 0.18000000000000071
Epoch: 607, Loss: 0.18000000000000071
Epoch: 608, Loss: 0.18000000000000071
Epoch: 609, Loss: 0.18000000000000071
Epoch: 610, Loss: 0.18000000000000071
Epoch: 611, Loss: 0.18000000000000071
Epoch: 612, Loss: 0.18000000000000071
Epoch: 613, Loss: 0.18000000000000071
Epoch: 614, Loss: 0.18000000000000071
Epoch: 615, Loss: 0.18000000000000071
Epoch: 616, Loss: 0.18000000000000071
Epoch: 617, Loss: 0.18000000000000071
Epoch: 618, Loss: 0.18000000000000071
Epoch: 619, Loss: 0.18000000000000071
Epoch: 620, Loss: 0.18000000000000071
Epoch: 621, Loss: 0.18000000000000071
Epoch: 622, Loss: 0.18000000000000071
Epoch: 623, Loss: 0.18000000000000071
Epoch: 624, Loss: 0.18000000000000071
Epoch: 625, Loss: 0.18000000000000071
Epoch: 626, Loss: 0.18000000000000071
Epoch: 627, Loss: 0.18000000000000071
Epoch: 628, Loss: 0.18000000000000071
Epoch: 629, Loss: 0.18000000000000071
Epoch: 630, Loss: 0.18000000000000071
Epoch: 631, Loss: 0.18000000000000071
Epoch: 632, Loss: 0.18000000000000071
Epoch: 633, Loss: 0.18000000000000071
Epoch: 634, Loss: 0.18000000000000071
Epoch: 635, Loss: 0.18000000000000071
Epoch: 636, Loss: 0.18000000000000071
Epoch: 637, Loss: 0.18000000000000071
Epoch: 638, Loss: 0.18000000000000071
Epoch: 639, Loss: 0.18000000000000071
Epoch: 640, Loss: 0.18000000000000071
Epoch: 641, Loss: 0.18000000000000071
Epoch: 642, Loss: 0.18000000000000071
Epoch: 643, Loss: 0.18000000000000071
Epoch: 644, Loss: 0.18000000000000071
Epoch: 645, Loss: 0.18000000000000071
Epoch: 646, Loss: 0.18000000000000071
Epoch: 647, Loss: 0.18000000000000071
Epoch: 648, Loss: 0.18000000000000071
Epoch: 649, Loss: 0.18000000000000071
Epoch: 650, Loss: 0.18000000000000071
Epoch: 651, Loss: 0.18000000000000071
Epoch: 652, Loss: 0.18000000000000071
Epoch: 653, Loss: 0.18000000000000071
Epoch: 654, Loss: 0.18000000000000071
Epoch: 655, Loss: 0.18000000000000071
Epoch: 656, Loss: 0.18000000000000071
Epoch: 657, Loss: 0.18000000000000071
Epoch: 658, Loss: 0.18000000000000071
Epoch: 659, Loss: 0.18000000000000071
Epoch: 660, Loss: 0.18000000000000071
Epoch: 661, Loss: 0.18000000000000071
Epoch: 662, Loss: 0.18000000000000071
Epoch: 663, Loss: 0.18000000000000071
Epoch: 664, Loss: 0.18000000000000071
Epoch: 665, Loss: 0.18000000000000071
Epoch: 666, Loss: 0.18000000000000071
Epoch: 667, Loss: 0.18000000000000071
Epoch: 668, Loss: 0.18000000000000071
Epoch: 669, Loss: 0.18000000000000071
Epoch: 670, Loss: 0.18000000000000071
Epoch: 671, Loss: 0.18000000000000071
Epoch: 672, Loss: 0.18000000000000071
Epoch: 673, Loss: 0.18000000000000071
Epoch: 674, Loss: 0.18000000000000071
Epoch: 675, Loss: 0.18000000000000071
Epoch: 676, Loss: 0.18000000000000071
Epoch: 677, Loss: 0.18000000000000071
Epoch: 678, Loss: 0.18000000000000055
Epoch: 679, Loss: 0.18000000000000077
Epoch: 680, Loss: 0.18000000000000066
Epoch: 681, Loss: 0.18000000000000066
Epoch: 682, Loss: 0.1800000000000008
Epoch: 683, Loss: 0.18000000000000066
Epoch: 684, Loss: 0.18000000000000066
Epoch: 685, Loss: 0.18000000000000066
Epoch: 686, Loss: 0.18000000000000066
Epoch: 687, Loss: 0.18000000000000066
Epoch: 688, Loss: 0.18000000000000066
Epoch: 689, Loss: 0.18000000000000066
Epoch: 690, Loss: 0.18000000000000066
Epoch: 691, Loss: 0.18000000000000066
Epoch: 692, Loss: 0.18000000000000066
Epoch: 693, Loss: 0.18000000000000066
Epoch: 694, Loss: 0.18000000000000066
Epoch: 695, Loss: 0.18000000000000066
Epoch: 696, Loss: 0.18000000000000066
Epoch: 697, Loss: 0.18000000000000066
Epoch: 698, Loss: 0.18000000000000066
Epoch: 699, Loss: 0.18000000000000066
Epoch: 700, Loss: 0.18000000000000066
Epoch: 701, Loss: 0.18000000000000066
Epoch: 702, Loss: 0.18000000000000066
Epoch: 703, Loss: 0.18000000000000066
Epoch: 704, Loss: 0.18000000000000066
Epoch: 705, Loss: 0.18000000000000066
Epoch: 706, Loss: 0.18000000000000066
Epoch: 707, Loss: 0.18000000000000066
Epoch: 708, Loss: 0.18000000000000066
Epoch: 709, Loss: 0.18000000000000066
Epoch: 710, Loss: 0.18000000000000066
Epoch: 711, Loss: 0.18000000000000066
Epoch: 712, Loss: 0.18000000000000066
Epoch: 713, Loss: 0.18000000000000066
Epoch: 714, Loss: 0.18000000000000066
Epoch: 715, Loss: 0.18000000000000066
Epoch: 716, Loss: 0.18000000000000066
Epoch: 717, Loss: 0.18000000000000066
Epoch: 718, Loss: 0.18000000000000066
Epoch: 719, Loss: 0.18000000000000066
Epoch: 720, Loss: 0.18000000000000066
Epoch: 721, Loss: 0.18000000000000066
Epoch: 722, Loss: 0.18000000000000066
Epoch: 723, Loss: 0.18000000000000066
Epoch: 724, Loss: 0.18000000000000066
Epoch: 725, Loss: 0.18000000000000066
Epoch: 726, Loss: 0.18000000000000066
Epoch: 727, Loss: 0.18000000000000066
Epoch: 728, Loss: 0.18000000000000066
Epoch: 729, Loss: 0.18000000000000066
Epoch: 730, Loss: 0.18000000000000066
Epoch: 731, Loss: 0.18000000000000066
Epoch: 732, Loss: 0.18000000000000066
Epoch: 733, Loss: 0.18000000000000066
Epoch: 734, Loss: 0.18000000000000066
Epoch: 735, Loss: 0.18000000000000066
Epoch: 736, Loss: 0.18000000000000066
Epoch: 737, Loss: 0.18000000000000066
Epoch: 738, Loss: 0.18000000000000066
Epoch: 739, Loss: 0.18000000000000066
Epoch: 740, Loss: 0.18000000000000066
Epoch: 741, Loss: 0.18000000000000066
Epoch: 742, Loss: 0.18000000000000066
Epoch: 743, Loss: 0.18000000000000066
Epoch: 744, Loss: 0.18000000000000066
Epoch: 745, Loss: 0.18000000000000066
Epoch: 746, Loss: 0.18000000000000066
Epoch: 747, Loss: 0.18000000000000066
Epoch: 748, Loss: 0.18000000000000066
Epoch: 749, Loss: 0.18000000000000066
Epoch: 750, Loss: 0.18000000000000066
Epoch: 751, Loss: 0.18000000000000066
Epoch: 752, Loss: 0.18000000000000066
Epoch: 753, Loss: 0.18000000000000066
Epoch: 754, Loss: 0.18000000000000066
Epoch: 755, Loss: 0.18000000000000066
Epoch: 756, Loss: 0.18000000000000066
Epoch: 757, Loss: 0.18000000000000066
Epoch: 758, Loss: 0.18000000000000066
Epoch: 759, Loss: 0.1800000000000005
Epoch: 760, Loss: 0.18000000000000071
Epoch: 761, Loss: 0.1800000000000005
Epoch: 762, Loss: 0.18000000000000071
Epoch: 763, Loss: 0.1800000000000005
Epoch: 764, Loss: 0.18000000000000071
Epoch: 765, Loss: 0.1800000000000005
Epoch: 766, Loss: 0.18000000000000083
Epoch: 767, Loss: 0.1800000000000005
Epoch: 768, Loss: 0.1800000000000005
Epoch: 769, Loss: 0.18000000000000083
Epoch: 770, Loss: 0.1800000000000005
Epoch: 771, Loss: 0.1800000000000005
Epoch: 772, Loss: 0.18000000000000083
Epoch: 773, Loss: 0.1800000000000004
Epoch: 774, Loss: 0.1800000000000007
Epoch: 775, Loss: 0.1800000000000007
Epoch: 776, Loss: 0.1800000000000007
Epoch: 777, Loss: 0.1800000000000007
Epoch: 778, Loss: 0.1800000000000007
Epoch: 779, Loss: 0.1800000000000007
Epoch: 780, Loss: 0.1800000000000007
Epoch: 781, Loss: 0.1800000000000007
Epoch: 782, Loss: 0.1800000000000007
Epoch: 783, Loss: 0.1800000000000007
Epoch: 784, Loss: 0.18000000000000052
Epoch: 785, Loss: 0.18000000000000038
Epoch: 786, Loss: 0.18000000000000022
Epoch: 787, Loss: 0.18000000000000027
Epoch: 788, Loss: 0.18000000000000022
Epoch: 789, Loss: 0.1800000000000001
Epoch: 790, Loss: 0.18000000000000044
Epoch: 791, Loss: 0.1800000000000004
Epoch: 792, Loss: 0.1800000000000001
Epoch: 793, Loss: 0.18000000000000044
Epoch: 794, Loss: 0.18000000000000044
Epoch: 795, Loss: 0.1800000000000005
Epoch: 796, Loss: 0.18000000000000022
Epoch: 797, Loss: 0.18000000000000022
Epoch: 798, Loss: 0.18000000000000038
Epoch: 799, Loss: 0.1800000000000002
Epoch: 800, Loss: 0.18000000000000052
Epoch: 801, Loss: 0.18000000000000024
Epoch: 802, Loss: 0.18000000000000055
Epoch: 803, Loss: 0.18000000000000022
Epoch: 804, Loss: 0.18000000000000055
Epoch: 805, Loss: 0.18000000000000016
Epoch: 806, Loss: 0.18000000000000024
Epoch: 807, Loss: 0.18000000000000055
Epoch: 808, Loss: 0.18000000000000022
Epoch: 809, Loss: 0.18000000000000055
Epoch: 810, Loss: 0.1800000000000002
Epoch: 811, Loss: 0.18000000000000022
Epoch: 812, Loss: 0.18000000000000055
Epoch: 813, Loss: 0.1800000000000002
Epoch: 814, Loss: 0.18000000000000022
Epoch: 815, Loss: 0.18000000000000044
Epoch: 816, Loss: 0.18000000000000016
Epoch: 817, Loss: 0.18000000000000024
Epoch: 818, Loss: 0.18000000000000024
Epoch: 819, Loss: 0.1800000000000002
Epoch: 820, Loss: 0.18000000000000024
Epoch: 821, Loss: 0.1800000000000002
Epoch: 822, Loss: 0.18000000000000024
Epoch: 823, Loss: 0.1800000000000002
Epoch: 824, Loss: 0.18000000000000024
Epoch: 825, Loss: 0.1800000000000002
Epoch: 826, Loss: 0.18000000000000024
Epoch: 827, Loss: 0.1800000000000002
Epoch: 828, Loss: 0.18000000000000024
Epoch: 829, Loss: 0.1800000000000002
Epoch: 830, Loss: 0.18000000000000024
Epoch: 831, Loss: 0.1800000000000002
Epoch: 832, Loss: 0.18000000000000024
Epoch: 833, Loss: 0.1800000000000002
Epoch: 834, Loss: 0.18000000000000024
Epoch: 835, Loss: 0.1800000000000002
Epoch: 836, Loss: 0.18000000000000024
Epoch: 837, Loss: 0.1800000000000002
Epoch: 838, Loss: 0.18000000000000024
Epoch: 839, Loss: 0.1800000000000002
Epoch: 840, Loss: 0.18000000000000024
Epoch: 841, Loss: 0.1800000000000002
Epoch: 842, Loss: 0.18000000000000024
Epoch: 843, Loss: 0.1800000000000002
Epoch: 844, Loss: 0.18000000000000024
Epoch: 845, Loss: 0.1800000000000002
Epoch: 846, Loss: 0.18000000000000024
Epoch: 847, Loss: 0.1800000000000002
Epoch: 848, Loss: 0.18000000000000024
Epoch: 849, Loss: 0.1800000000000002
Epoch: 850, Loss: 0.18000000000000024
Epoch: 851, Loss: 0.1800000000000002
Epoch: 852, Loss: 0.18000000000000024
Epoch: 853, Loss: 0.1800000000000002
Epoch: 854, Loss: 0.18000000000000024
Epoch: 855, Loss: 0.1800000000000002
Epoch: 856, Loss: 0.18000000000000024
Epoch: 857, Loss: 0.1800000000000002
Epoch: 858, Loss: 0.18000000000000024
Epoch: 859, Loss: 0.1800000000000002
Epoch: 860, Loss: 0.18000000000000024
Epoch: 861, Loss: 0.1800000000000002
Epoch: 862, Loss: 0.18000000000000024
Epoch: 863, Loss: 0.1800000000000002
Epoch: 864, Loss: 0.18000000000000024
Epoch: 865, Loss: 0.1800000000000002
Epoch: 866, Loss: 0.18000000000000024
Epoch: 867, Loss: 0.1800000000000002
Epoch: 868, Loss: 0.18000000000000024
Epoch: 869, Loss: 0.1800000000000002
Epoch: 870, Loss: 0.18000000000000024
Epoch: 871, Loss: 0.1800000000000002
Epoch: 872, Loss: 0.18000000000000024
Epoch: 873, Loss: 0.1800000000000002
Epoch: 874, Loss: 0.18000000000000024
Epoch: 875, Loss: 0.1800000000000002
Epoch: 876, Loss: 0.18000000000000024
Epoch: 877, Loss: 0.1800000000000002
Epoch: 878, Loss: 0.18000000000000024
Epoch: 879, Loss: 0.1800000000000002
Epoch: 880, Loss: 0.18000000000000024
Epoch: 881, Loss: 0.1800000000000002
Epoch: 882, Loss: 0.18000000000000024
Epoch: 883, Loss: 0.1800000000000002
Epoch: 884, Loss: 0.18000000000000024
Epoch: 885, Loss: 0.1800000000000002
Epoch: 886, Loss: 0.18000000000000024
Epoch: 887, Loss: 0.1800000000000002
Epoch: 888, Loss: 0.18000000000000024
Epoch: 889, Loss: 0.1800000000000002
Epoch: 890, Loss: 0.18000000000000024
Epoch: 891, Loss: 0.1800000000000002
Epoch: 892, Loss: 0.18000000000000024
Epoch: 893, Loss: 0.1800000000000002
Epoch: 894, Loss: 0.18000000000000024
Epoch: 895, Loss: 0.1800000000000002
Epoch: 896, Loss: 0.18000000000000024
Epoch: 897, Loss: 0.1800000000000002
Epoch: 898, Loss: 0.18000000000000024
Epoch: 899, Loss: 0.1800000000000002
Epoch: 900, Loss: 0.18000000000000024
Epoch: 901, Loss: 0.1800000000000002
Epoch: 902, Loss: 0.18000000000000024
Epoch: 903, Loss: 0.1800000000000002
Epoch: 904, Loss: 0.18000000000000024
Epoch: 905, Loss: 0.1800000000000002
Epoch: 906, Loss: 0.18000000000000024
Epoch: 907, Loss: 0.1800000000000002
Epoch: 908, Loss: 0.18000000000000024
Epoch: 909, Loss: 0.1800000000000002
Epoch: 910, Loss: 0.18000000000000024
Epoch: 911, Loss: 0.1800000000000002
Epoch: 912, Loss: 0.18000000000000024
Epoch: 913, Loss: 0.1800000000000002
Epoch: 914, Loss: 0.18000000000000024
Epoch: 915, Loss: 0.1800000000000002
Epoch: 916, Loss: 0.18000000000000024
Epoch: 917, Loss: 0.1800000000000002
Epoch: 918, Loss: 0.18000000000000024
Epoch: 919, Loss: 0.1800000000000002
Epoch: 920, Loss: 0.18000000000000024
Epoch: 921, Loss: 0.1800000000000002
Epoch: 922, Loss: 0.18000000000000024
Epoch: 923, Loss: 0.1800000000000002
Epoch: 924, Loss: 0.18000000000000024
Epoch: 925, Loss: 0.1800000000000002
Epoch: 926, Loss: 0.18000000000000024
Epoch: 927, Loss: 0.1800000000000002
Epoch: 928, Loss: 0.18000000000000024
Epoch: 929, Loss: 0.1800000000000002
Epoch: 930, Loss: 0.18000000000000024
Epoch: 931, Loss: 0.1800000000000002
Epoch: 932, Loss: 0.18000000000000024
Epoch: 933, Loss: 0.1800000000000002
Epoch: 934, Loss: 0.18000000000000024
Epoch: 935, Loss: 0.1800000000000002
Epoch: 936, Loss: 0.18000000000000024
Epoch: 937, Loss: 0.1800000000000002
Epoch: 938, Loss: 0.18000000000000024
Epoch: 939, Loss: 0.1800000000000002
Epoch: 940, Loss: 0.18000000000000024
Epoch: 941, Loss: 0.1800000000000002
Epoch: 942, Loss: 0.18000000000000024
Epoch: 943, Loss: 0.1800000000000002
Epoch: 944, Loss: 0.18000000000000024
Epoch: 945, Loss: 0.1800000000000002
Epoch: 946, Loss: 0.18000000000000024
Epoch: 947, Loss: 0.1800000000000002
Epoch: 948, Loss: 0.18000000000000024
Epoch: 949, Loss: 0.1800000000000002
Epoch: 950, Loss: 0.18000000000000024
Epoch: 951, Loss: 0.1800000000000002
Epoch: 952, Loss: 0.18000000000000024
Epoch: 953, Loss: 0.1800000000000002
Epoch: 954, Loss: 0.18000000000000024
Epoch: 955, Loss: 0.1800000000000002
Epoch: 956, Loss: 0.18000000000000024
Epoch: 957, Loss: 0.1800000000000002
Epoch: 958, Loss: 0.18000000000000024
Epoch: 959, Loss: 0.1800000000000002
Epoch: 960, Loss: 0.18000000000000024
Epoch: 961, Loss: 0.1800000000000002
Epoch: 962, Loss: 0.18000000000000024
Epoch: 963, Loss: 0.1800000000000002
Epoch: 964, Loss: 0.18000000000000024
Epoch: 965, Loss: 0.1800000000000002
Epoch: 966, Loss: 0.18000000000000024
Epoch: 967, Loss: 0.1800000000000002
Epoch: 968, Loss: 0.18000000000000024
Epoch: 969, Loss: 0.1800000000000002
Epoch: 970, Loss: 0.18000000000000024
Epoch: 971, Loss: 0.1800000000000002
Epoch: 972, Loss: 0.18000000000000024
Epoch: 973, Loss: 0.1800000000000002
Epoch: 974, Loss: 0.18000000000000024
Epoch: 975, Loss: 0.1800000000000002
Epoch: 976, Loss: 0.18000000000000024
Epoch: 977, Loss: 0.1800000000000002
Epoch: 978, Loss: 0.18000000000000024
Epoch: 979, Loss: 0.1800000000000002
Epoch: 980, Loss: 0.18000000000000024
Epoch: 981, Loss: 0.1800000000000002
Epoch: 982, Loss: 0.18000000000000016
Epoch: 983, Loss: 0.1800000000000002
Epoch: 984, Loss: 0.18000000000000013
Epoch: 985, Loss: 0.1800000000000002
Epoch: 986, Loss: 0.18000000000000013
Epoch: 987, Loss: 0.1800000000000002
Epoch: 988, Loss: 0.18000000000000013
Epoch: 989, Loss: 0.1800000000000002
Epoch: 990, Loss: 0.18000000000000013
Epoch: 991, Loss: 0.1800000000000002
Epoch: 992, Loss: 0.18000000000000005
Epoch: 993, Loss: 0.18000000000000005
Epoch: 994, Loss: 0.18000000000000005
Epoch: 995, Loss: 0.18000000000000005
Epoch: 996, Loss: 0.18000000000000033
Epoch: 997, Loss: 0.18000000000000013
Epoch: 998, Loss: 0.18000000000000002
Epoch: 999, Loss: 0.18
(.venv) kyleuphus@Kyles-MacBook-Pro-4 ml-library %   
