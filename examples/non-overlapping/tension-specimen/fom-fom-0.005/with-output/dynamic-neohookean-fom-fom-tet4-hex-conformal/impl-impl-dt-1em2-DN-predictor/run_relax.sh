echo "Running relax = 0.1 case..."
cd relax0_1 
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.2 case..."
cd ../relax0_2 
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.3 case..."
cd ../relax0_3
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.4 case..."
cd ../relax0_4
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.5 case..."
cd ../relax0_5 
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.6 case..."
cd ../relax0_6 
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.7 case..."
cd ../relax0_7 
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.8 case..."
cd ../relax0_8
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.9 case..."
cd ../relax0_9
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
echo "Running relax = 1 case..."
cd ../relax1 
julia --project=@/home/ikalash/Norma.jl.votd /home/ikalash/Norma.jl.votd/src/Norma.jl tension-specimen.yaml >& out.txt 
echo "...done." 
cd ../
