// Deterministic executable for pipe/timeout tests; never invokes Windows WSL.
using System;
using System.Text;
using System.Threading;

class WslProbeFixture
{
    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.Unicode;
        if (args.Length == 1 && args[0] == "--version")
        {
            Console.WriteLine("WSL version: 2.6.1.0");
            return 0;
        }
        if (args.Length == 2 && args[1] == "--quiet")
        {
            // Each pipe exceeds its buffer. Sequential draining would deadlock.
            Console.Out.Write(new string('o', 131072));
            Console.Error.Write(new string('e', 131072));
            return 0;
        }
        if (args.Length == 2 && args[1] == "--verbose")
        {
            Thread.Sleep(10000);
            return 0;
        }
        return 5;
    }
}
