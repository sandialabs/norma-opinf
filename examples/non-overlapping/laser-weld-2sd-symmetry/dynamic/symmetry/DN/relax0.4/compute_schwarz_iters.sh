

grep "Performed" out.txt >& schwarz_iters.txt
sed -i "s/\[SCHWARZ\] Performed//g" schwarz_iters.txt
sed -i "s/Schwarz Iterations//g" schwarz_iters.txt
sed -i "s/Schwarz Iteration//g" schwarz_iters.txt
awk '{ sum += $1; count++ } END { if (count > 0) print sum / count; else print "No numbers found." }' schwarz_iters.txt >> tmp.out
sort schwarz_iters.txt >& si.txt
sort -n si.txt  >& si2.txt 
tail -n1 si2.txt >> tmp.out
rm si.txt
cat tmp.out

