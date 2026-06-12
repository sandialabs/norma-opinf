
grep "Performed" out.txt >& schwarz_iters.txt
sed -i "s/\[SCHWARZ\] Performed//g" schwarz_iters.txt
sed -i "s/Schwarz Iterations//g" schwarz_iters.txt
sed -i "s/Schwarz Iteration//g" schwarz_iters.txt
awk '{ sum += $1; count++ } END { if (count > 0) print sum / count; else print "No numbers found." }' schwarz_iters.txt >> tmp.out
sort -n schwarz_iters.txt >& si.txt
tail -n1 si.txt >> tmp.out
rm si.txt
cat tmp.out
