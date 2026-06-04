echo "Running relax = 0.1 case..."
cd relax0.1 
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
#echo "Running relax = 0.2 case..."
#cd ../relax0.2 
#julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
#echo "...done." 
echo "Running relax = 0.3 case..."
cd ../relax0.3
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.4 case..."
cd ../relax0.4
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.5 case..."
cd ../relax0.5 
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.6 case..."
cd ../relax0.6 
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.7 case..."
cd ../relax0.7 
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.8 case..."
cd ../relax0.8
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 0.9 case..."
cd ../relax0.9
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
echo "Running relax = 1 case..."
cd ../relax1 
julia --project=@/home/ikalash/Norma.jl.votd3 /home/ikalash/Norma.jl.votd3/src/Norma.jl bracket.yaml >& out.txt 
echo "...done." 
cd ../
