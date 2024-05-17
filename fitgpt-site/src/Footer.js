function Footer() {
	return (
		<footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t bg-gray-950 text-gray-50">
			<p className="text-xs text-gray-400">© 2024 AI Image Classifier. All rights reserved.</p>
			<nav className="sm:ml-auto flex gap-4 sm:gap-6">
				<a href="/" className="text-xs hover:underline underline-offset-4 text-gray-400">Terms of Service</a>
				<a href="/" className="text-xs hover:underline underline-offset-4 text-gray-400">Privacy</a>
			</nav>
		</footer>
	)
}

export default Footer;
