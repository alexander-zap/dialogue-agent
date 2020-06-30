def run(orig, dest):
    outsize = 0
    with open(orig, 'rb') as infile:
        content = infile.read()
    with open(dest, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

    print("Done. Saved %s bytes." % (len(content) - outsize))


original = "resources/movie_db.pkl"
destination = "resources/movie_db.pkl"
run(original, destination)
