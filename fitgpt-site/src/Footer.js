function Footer() {
	return (
		<footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t bg-gray-950 text-gray-50">
			<p className="text-xs text-gray-400">The FitGPT Team.</p>
			<nav className="sm:ml-auto flex gap-4 sm:gap-6">
				<a href="https://github.com/qwertyblurby/FitGPT" className="text-xs hover:underline underline-offset-4 text-gray-400">Our GitHub</a>
			</nav>
		</footer>
	)
}

export default Footer;
