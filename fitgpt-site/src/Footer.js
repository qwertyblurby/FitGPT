import GitHubIcon from './assets/icons/GitHubIcon';

function Footer() {
	return (
		<footer className="flex flex-col sm:flex-row sm:justify-between py-6 w-full shrink-0 items-center px-4 md:px-6 border-t bg-gray-950 text-gray-50">
			<p className="text-xs text-gray-400">The FitGPT Team.</p>
			<a href="https://github.com/qwertyblurby/FitGPT" className="text-xs hover:underline underline-offset-4 text-gray-400">
				<GitHubIcon isProfile={true}/>Our GitHub
			</a>
		</footer>
	)
}

export default Footer;
