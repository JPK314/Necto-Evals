Set-Location -Path 'E:\User\Documents\GitHub\mELO-necto-evals'

for ($num = 1 ; $num -le 7 ; $num++){
    Start-Process cmd "/c `"C:\Users\User\miniconda3\envs\rlgym\python.exe main.py outputs/out_$num.txt & pause `""
}